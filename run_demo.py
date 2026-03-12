
import os
import trimesh
import yaml
import numpy as np
import cv2
import torch

from PIL import Image
from estimater import Any6D
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing
from pytorch3d.renderer import (
    BlendParams,
    HardPhongShader,
    MeshRasterizer,
    PointLights,
    RasterizationSettings,
    PerspectiveCameras,
    TexturesUV,
    TexturesVertex,
)
from pytorch3d.structures import Meshes

try:
    from pytorch3d.utils import cameras_from_opencv_projection
except Exception:
    cameras_from_opencv_projection = None

from foundationpose.Utils import get_bounding_box, visualize_frame_results, calculate_chamfer_distance_gt_mesh, align_mesh_to_coordinate, make_mesh_tensors
import nvdiffrast.torch as dr
import argparse
from pytorch_lightning import seed_everything

from sam2_instantmesh import *

glctx = dr.RasterizeCudaContext()


def _project_rotation_to_so3(rotation_mat):
    u, _, v_t = torch.linalg.svd(rotation_mat)
    r = u @ v_t
    if torch.det(r) < 0:
        u = u.clone()
        u[:, -1] *= -1.0
        r = u @ v_t
    return r


def _build_pytorch3d_cameras(K, R, t, image_h, image_w, device):
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device)[None]
    image_size = torch.tensor([[float(image_h), float(image_w)]], dtype=torch.float32, device=device)
    R_t = R[None]
    t_t = t[None]

    if cameras_from_opencv_projection is not None:
        return cameras_from_opencv_projection(
            R=R_t,
            tvec=t_t,
            camera_matrix=K_t,
            image_size=image_size,
        )

    axis_flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device, dtype=torch.float32))
    R_p3d = axis_flip[None] @ R_t
    t_p3d = t_t.clone()
    t_p3d[:, :2] *= -1.0

    focal_length = torch.stack((K_t[:, 0, 0], K_t[:, 1, 1]), dim=1)
    principal_point = torch.stack((K_t[:, 0, 2], K_t[:, 1, 2]), dim=1)
    return PerspectiveCameras(
        R=R_p3d,
        T=t_p3d,
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=image_size,
        in_ndc=False,
        device=device,
    )


def _build_mesh_textures(mesh, vertices_param, faces, device):
    visual = getattr(mesh, "visual", None)
    if (
        visual is not None
        and hasattr(visual, "uv")
        and visual.uv is not None
        and hasattr(visual, "material")
        and visual.material is not None
        and hasattr(visual.material, "image")
        and visual.material.image is not None
    ):
        tex_img = np.array(visual.material.image.convert("RGB"), dtype=np.float32) / 255.0
        tex_map = torch.as_tensor(tex_img, dtype=torch.float32, device=device)[None]
        verts_uvs = torch.as_tensor(visual.uv.copy(), dtype=torch.float32, device=device)
        verts_uvs[:, 1] = 1.0 - verts_uvs[:, 1]
        faces_uvs = faces[None].long()
        return TexturesUV(maps=tex_map, faces_uvs=faces_uvs, verts_uvs=verts_uvs[None])

    if visual is not None and hasattr(visual, "vertex_colors") and visual.vertex_colors is not None:
        vertex_colors = visual.vertex_colors[:, :3]
        if len(vertex_colors) == vertices_param.shape[0]:
            verts_rgb = torch.as_tensor(vertex_colors, dtype=torch.float32, device=device) / 255.0
            return TexturesVertex(verts_features=verts_rgb[None])

    default_rgb = torch.full((1, vertices_param.shape[0], 3), 0.7, dtype=torch.float32, device=device)
    return TexturesVertex(verts_features=default_rgb)


def refine_shape_and_pose(
    input_image,
    K,
    mesh,
    init_pose,
    object_mask=None,
    outer_iters=3,
    pose_steps=10,
    shape_steps=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_h, image_w = input_image.shape[:2]

    target_rgb = torch.as_tensor(input_image[..., :3], dtype=torch.float32, device=device)[None] / 255.0
    mask_t = None
    if object_mask is not None:
        mask_t = torch.as_tensor(object_mask.astype(np.float32), dtype=torch.float32, device=device)[None]

    V_init = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(mesh.faces, dtype=torch.int64, device=device)
    V = torch.nn.Parameter(V_init.clone())
    V.requires_grad_(True)

    init_pose_t = torch.as_tensor(init_pose, dtype=torch.float32, device=device)
    R = torch.nn.Parameter(init_pose_t[:3, :3].clone())
    t = torch.nn.Parameter(init_pose_t[:3, 3].clone())
    R.requires_grad_(True)
    t.requires_grad_(True)

    textures = _build_mesh_textures(mesh, V, faces, device)

    raster_settings = RasterizationSettings(
        image_size=(image_h, image_w),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 0.0]],
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((0.0, 0.0, 0.0),),
        specular_color=((0.0, 0.0, 0.0),),
    )

    optimizer_T = torch.optim.Adam([R, t], lr=1e-3)
    optimizer_V = torch.optim.Adam([V], lr=1e-4)

    def render_and_photometric_loss():
        cameras = _build_pytorch3d_cameras(K=K, R=R, t=t, image_h=image_h, image_w=image_w, device=device)
        meshes = Meshes(verts=[V], faces=[faces], textures=textures)

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(meshes)
        shader = HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params,
        )
        rendered = shader(fragments, meshes, cameras=cameras, lights=lights)
        rendered_rgb = rendered[..., :3]

        visible = (fragments.pix_to_face[..., 0] >= 0).float()
        if mask_t is not None:
            loss_mask = visible * mask_t
        else:
            loss_mask = visible

        valid_mask_pixels = loss_mask.sum()
        if valid_mask_pixels <= 0:
            loss_mask = torch.ones_like(loss_mask)
            valid_mask_pixels = loss_mask.sum()

        photo_loss = (torch.abs(rendered_rgb - target_rgb) * loss_mask[..., None]).sum() / (valid_mask_pixels * 3.0 + 1e-6)
        return photo_loss, meshes

    for outer_iter in range(outer_iters):
        V.requires_grad_(False)
        R.requires_grad_(True)
        t.requires_grad_(True)

        pose_loss_val = None
        for _ in range(pose_steps):
            optimizer_T.zero_grad(set_to_none=True)
            pose_loss, _ = render_and_photometric_loss()
            pose_loss.backward()
            optimizer_T.step()
            with torch.no_grad():
                R.copy_(_project_rotation_to_so3(R))
            pose_loss_val = float(pose_loss.detach().item())

        V.requires_grad_(True)
        R.requires_grad_(False)
        t.requires_grad_(False)

        shape_loss_val = None
        for _ in range(shape_steps):
            optimizer_V.zero_grad(set_to_none=True)
            photometric_loss, meshes = render_and_photometric_loss()
            laplacian_loss = mesh_laplacian_smoothing(meshes, method="uniform")
            edge_loss = mesh_edge_loss(meshes)
            total_shape_loss = photometric_loss + 0.1 * laplacian_loss + 0.01 * edge_loss
            total_shape_loss.backward()
            optimizer_V.step()
            shape_loss_val = float(total_shape_loss.detach().item())

        print(
            f"[Alt-Refine] outer {outer_iter + 1}/{outer_iters} "
            f"pose_loss={pose_loss_val:.6f} shape_loss={shape_loss_val:.6f}"
        )

    with torch.no_grad():
        R_final = _project_rotation_to_so3(R)
        refined_pose = torch.eye(4, dtype=torch.float32, device=device)
        refined_pose[:3, :3] = R_final
        refined_pose[:3, 3] = t
        refined_vertices = V.detach().cpu().numpy()

    return refined_pose.detach().cpu().numpy(), refined_vertices

if __name__=='__main__':

    seed_everything(0)

    parser = argparse.ArgumentParser(description="Set experiment name and paths")
    parser.add_argument("--ycb_model_path", type=str, default="/workspace/datasets/Pose_datasets/ho3d/YCB_Video_Models", help="Path to the YCB Video Models")
    parser.add_argument("--img_to_3d", action="store_true",help="Running with InstantMesh+SAM2")
    args = parser.parse_args()


    ycb_model_path = args.ycb_model_path
    img_to_3d = args.img_to_3d

    results = []
    demo_path = 'demo_data'
    mesh_path = os.path.join(demo_path, f'mustard.obj')

    obj = 'demo_mustard'
    save_path = f'results/{obj}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    depth_scale = 1000.0
    color = cv2.cvtColor(cv2.imread(os.path.join(demo_path, 'color.png')), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(demo_path, 'depth.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32) / depth_scale
    Image.fromarray(color).save(os.path.join(save_path, 'color.png'))

    label = np.load(os.path.join(demo_path, 'labels.npz'))
    obj_num = 5
    mask = np.where(label['seg'] == obj_num, 255, 0).astype(np.bool_)

    if img_to_3d:
        cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
        input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
        mask_refine = running_sam_box(color, input_box)

        input_image = preprocess_image(color, mask_refine, save_path, obj)
        images = diffusion_image_generation(save_path, save_path, obj, input_image=input_image)
        instant_mesh_process(images, save_path, obj)

        mesh = trimesh.load(os.path.join(save_path, f'mesh_{obj}.obj'))
        mesh = align_mesh_to_coordinate(mesh)
        mesh.export(os.path.join(save_path, f'center_mesh_{obj}.obj'))

        mesh = trimesh.load(os.path.join(save_path, f'center_mesh_{obj}.obj'))
    else:
        mesh = trimesh.load(mesh_path)


    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=2)

    # camera info
    intrinsic_path = f"{demo_path}/836212060125_640x480.yml"
    with open(intrinsic_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    intrinsic = np.array([[data["depth"]["fx"], 0.0, data["depth"]["ppx"]], [0.0, data["depth"]["fy"], data["depth"]["ppy"]], [0.0, 0.0, 1.0], ], )
    np.savetxt(os.path.join(save_path, f'K.txt'), intrinsic)

    pred_pose = est.register_any6d(K=intrinsic, rgb=color, depth=depth, ob_mask=mask, iteration=5, name=f'demo')

    pose_centered_init = est.pose_last.detach().cpu().numpy() if torch.is_tensor(est.pose_last) else pred_pose.copy()
    pose_centered_refined, refined_vertices = refine_shape_and_pose(
        input_image=color,
        K=intrinsic,
        mesh=est.mesh,
        init_pose=pose_centered_init,
        object_mask=mask,
    )

    est.mesh.vertices = refined_vertices
    est.mesh_tensors = make_mesh_tensors(est.mesh)
    pose_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    est.pose_last = torch.as_tensor(pose_centered_refined, dtype=torch.float32, device=pose_device)

    tf_to_centered = np.eye(4, dtype=np.float32)
    tf_to_centered[:3, 3] = -est.model_center
    pred_pose = pose_centered_refined @ tf_to_centered

    pose_list = label['pose_y']
    index_list = np.unique(label['seg'])
    index = (np.where(index_list == obj_num)[0] - 1).tolist()[0]
    tmp = pose_list[index]
    gt_pose = np.eye(4)
    gt_pose[:3, :] = tmp
    # import ipdb; ipdb.set_trace()
    gt_mesh = trimesh.load(f'{ycb_model_path}/025_mug/textured_simple.obj')

    chamfer_dis = calculate_chamfer_distance_gt_mesh(gt_pose, gt_mesh, pred_pose, est.mesh)
    print(chamfer_dis)

    np.savetxt(os.path.join(save_path, f'{obj}_initial_pose.txt'), pred_pose)
    np.savetxt(os.path.join(save_path, f'{obj}_gt_pose.txt'), gt_pose)
    est.mesh.export(os.path.join(save_path, f'final_mesh_{obj}.obj'))

    np.savetxt(os.path.join(save_path, f'{obj}_cd.txt'), [chamfer_dis])

    results.append({
        'Object': obj,
        'Object_Number': obj_num,
        'Chamfer_Distance': float(chamfer_dis)
        })



