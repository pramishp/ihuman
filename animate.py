import cv2
import hydra
import numpy as np
import torch
import imageio
import open3d as o3d
import pytorch_lightning as pl
import pickle
import joblib
import tqdm
import pytorch3d.transforms as T
from animatableGaussian.model.nerf_model import NeRFModel

DEVICE = "cuda"


def load_mixamo_smpl(actions_dir, action_type='0007', skip=1):
    result = joblib.load(os.path.join(actions_dir, action_type, 'result.pkl'))

    anim_len = result['anim_len']
    pose_array = result['smpl_array'].reshape(anim_len, -1)
    cam_array = result['cam_array']

    global_orients = []
    body_poses = []
    transls = []

    for i in range(0, anim_len, skip):
        global_orients.append(pose_array[i, :3].tolist())
        body_poses.append(pose_array[i, 3:72].tolist())
        transls.append(np.array([cam_array[i, 1], cam_array[i, 2], 0]).tolist())

    params = {
        'body_pose': torch.tensor(body_poses, device=DEVICE),
        'global_orient': torch.tensor(global_orients, device=DEVICE),
        'transl': torch.tensor(transls, device=DEVICE),
    }
    return params


def too3dmesh(vertx, triangles, colors=None):
    verts_posed_o3d = o3d.utility.Vector3dVector(vertx.detach().cpu().numpy())
    faces_o3d = o3d.utility.Vector3iVector(triangles.detach().cpu().numpy())
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = verts_posed_o3d
    mesh_o3d.triangles = faces_o3d
    if colors is not None:
        vertex_colors_o3d = o3d.utility.Vector3dVector(colors.detach().cpu().numpy())
        mesh_o3d.vertex_colors = vertex_colors_o3d
    return mesh_o3d


@hydra.main(config_path="./confs", config_name="mmpeoplesnapshot_fine", version_base="1.1")
def main(opt):
    output_path = "animations"
    pl.seed_everything(0)
    os.makedirs(output_path, exist_ok=True)
    checkpoint_path = "model.ckpt"
    model = NeRFModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.to('cuda')

    animations = []
    animation_poses = load_mixamo_smpl('../../data/animation/motion',
                                       action_type='0022', 
                                       )

    poses = animation_poses['body_pose']
    transl = animation_poses['transl']
    global_orients = animation_poses['global_orient']

    camera_path = "../../data/animation/camera.pkl"
    with open(camera_path, 'rb') as f:
        camera_params = pickle.load(f)

    for idx, (body_pose, transl, global_orient) in tqdm.tqdm(enumerate(zip(poses, transl, global_orients))):

        params = {
            'body_pose': body_pose.reshape(1, 23, 3).to('cuda'),
            'transl': transl.reshape(1, 3).to('cuda'),
            'global_orient': global_orient.reshape(1, 3).to('cuda')
        }


        global_orient = params['global_orient']
        angle = (torch.pi / 180) * (360/poses.shape[0]) * idx
        additional_rotation_matrix = torch.tensor([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], device='cuda:0').unsqueeze(0).float() 


        rot_global_orient_matrix = T.axis_angle_to_matrix(global_orient)
        combined_rot_matrix = torch.bmm(additional_rotation_matrix, rot_global_orient_matrix)
        combined_rot_angle_axis = T.matrix_to_axis_angle(combined_rot_matrix)

        params['global_orient'] = combined_rot_angle_axis

        params["transl"] += torch.tensor([0, 0.15, 5]).to('cuda')

        rgb, vt, scales, rotation, depth, alpha, normal = model(camera_params, params, 0, train=False)

        img_path = f"{output_path}/{idx}.png"
        image = rgb.detach().cpu().permute(1, 2, 0).numpy() * 255.0
        animations.append(image)
        cv2.imwrite(img_path, image[:, :, ::-1])

        verts_posed = vt
        pred_faces = model.model.faces[0]
        vertex_colors = model.model.get_vertex_colors()
        mesh_o3d = too3dmesh(verts_posed, pred_faces, vertex_colors)
        os.makedirs(f"{output_path}/mesh", exist_ok=True)
        o3d.io.write_triangle_mesh(f"{output_path}/mesh/{idx}.obj", mesh_o3d)
    
    animations = [np.asarray(animation, dtype=np.uint8) for animation in animations]   
    imageio.mimsave(f"{output_path}/training.gif", animations)


if __name__ == "__main__":
    import os
    main()
