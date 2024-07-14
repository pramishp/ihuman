import torch.nn as nn
from animatableGaussian.deformer.encoder.position_encoder import SHEncoder, DisplacementEncoder
import torch
import pickle
import os
import numpy as np
from animatableGaussian.deformer.lbs import lbs, vertices2joints
from simple_knn._C import distCUDA2

SMPL_TO_BODY25 = [
    24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 2, 28, 29,
    30, 31, 32, 33, 34
]


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class SMPLModel(nn.Module):
    def __init__(self, model_path, max_sh_degree=0, max_freq=4, gender="male", num_repeat=20, num_players=1,
                 use_point_color=False, use_point_displacement=False, model_size="small"):
        super().__init__()

        self.num_players = num_players
        self.use_point_displacement = use_point_displacement
        self.use_point_color = use_point_color

        smpl_path = os.path.join(model_path, model_size)

        v_template = np.loadtxt(os.path.join(
            smpl_path, 'vertices.txt'))
        v_template_center = v_template.mean(0)
        v_template = v_template - v_template_center
        weights = np.loadtxt(os.path.join(
            smpl_path, 'weights.txt'))
        kintree_table = np.loadtxt(os.path.join(
            smpl_path, 'kintree_table.txt'))

        J = np.loadtxt(os.path.join(
            smpl_path, 'joint_locations.txt'))
        J = J - v_template_center

        # add faces/triangles
        faces = np.loadtxt(os.path.join(smpl_path, 'triangles.txt')).astype(int)
        self.register_buffer('faces', torch.from_numpy(faces).unsqueeze(0))

        ########## ========== UVS ===============
        self.mesh2smpl_idx = np.loadtxt(os.path.join(smpl_path, 'mesh2smpl_idx.txt')).astype(int)

        verts_uv = np.loadtxt(os.path.join(smpl_path, 'uv_coords.txt')).astype(np.float32)
        verts_uv = torch.from_numpy(verts_uv).unsqueeze(0)

        self.verts2uv_idx_ = np.loadtxt(os.path.join(smpl_path, 'vertex_indices.txt')).astype(int)
        self.verts2uv_idx = np.arange(0, v_template.shape[0])

        from animatableGaussian.utils import many2one_mapper
        self.uv2verts_idx = many2one_mapper(self.verts2uv_idx)
        # UVs = np.loadtxt(os.path.join(smpl_path, 'smpl_uvs.txt'))
        self.register_buffer('uvs', verts_uv.repeat([self.num_players, 1, 1]))

        self.register_buffer('v_template', torch.Tensor(
            v_template)[None, ...].repeat(
            [self.num_players, 1, 1]))
        dist2 = torch.clamp_min(
            distCUDA2(self.v_template[0].cuda()), 0.0000001)[..., None].repeat([num_repeat, 3])
        dist2 /= num_repeat

        self.v_template = self.v_template.repeat([1, num_repeat, 1])
        ## pytorch 3d mesh
        from pytorch3d.structures import Meshes

        mesh = Meshes(verts=self.v_template, faces=self.faces)
        mesh._compute_vertex_normals()
        self.register_buffer("vertex_normals", mesh.verts_normals_packed())  # NX3

        # scales and rotation
        from animatableGaussian.deformer.init_helpers import init_qs_on_mesh
        quaternions, scaling = init_qs_on_mesh(mesh.verts_packed(),
                                               mesh.faces_packed(),
                                               mesh.verts_normals_packed(),
                                               scale_init_factor=0.1,
                                               max_scale=1.0,
                                               min_scale=0.0,
                                               thickness_init_factor=0.001)

        self.weights = nn.Parameter(
            torch.Tensor(weights[self.mesh2smpl_idx]).repeat([num_repeat, 1]))
        self.parents = kintree_table[0].astype(np.int64)
        self.parents[0] = -1

        self.J = nn.Parameter(torch.Tensor(
            J)[None, ...].repeat([self.num_players, 1, 1]))

        minmax = [self.v_template[0].min(
            dim=0).values * 1.05, self.v_template[0].max(dim=0).values * 1.05]
        self.register_buffer('normalized_vertices',
                             (self.v_template - minmax[0]) / (minmax[1] - minmax[0]))

        if use_point_displacement:
            self.displacements = nn.Parameter(
                torch.zeros_like(self.v_template))
        else:
            self.displacementEncoder = DisplacementEncoder(
                encoder="hash", num_players=num_players)

        n = self.v_template.shape[1] * num_players

        if use_point_color:
            self.shs_dc = nn.Parameter(torch.zeros(
                [n, 1, 3]))
            self.shs_rest = nn.Parameter(torch.zeros(
                [n, (max_sh_degree + 1) ** 2 - 1, 3]))
        else:
            self.shEncoder = SHEncoder(max_sh_degree=max_sh_degree,
                                       encoder="hash", num_players=num_players)
        # self.opacity = nn.Parameter(inverse_sigmoid(
        #     0.2 * torch.ones((n, 1), dtype=torch.float)))
        self.opacity = nn.Parameter(torch.ones((n, 1)) * 0.99)

        self.scales = nn.Parameter(
            (scaling.repeat([num_players, 1])))
        # rotations = torch.zeros([n, 4])
        # rotations[:, 0] = 1
        # self.rotations = nn.Parameter(rotations)
        self.rotations = nn.Parameter(quaternions)

        J_regressor = torch.tensor(np.loadtxt(os.path.join(smpl_path, 'J_regressor.txt')).astype(np.float32))
        self.register_buffer('J_regressor', J_regressor)

        # add model scale parameter
        self.model_scale = nn.Parameter(torch.tensor(1.0))

        ##### LAB
        import smplx

        self.body_model = smplx.SMPL("../../data/smpl/models/", gender='male').to('cuda')
        self.body_model.requires_grad_(False)

    def configure_optimizers(self, training_args, extra_params=None):
        l = [
            {'params': [self.weights],
             'lr': training_args.weights_lr, "name": "weights"},
            {'params': [self.J], 'lr': training_args.joint_lr, "name": "J"},
            {'params': [self.opacity],
             'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.scales],
             'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self.rotations],
             'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self.model_scale],
             'lr': 0,  # 1e-3
             "name": "model_scale"}  # TODO: set model scale lr in config
        ]
        if extra_params:
            l = [*l, *extra_params]

        if self.use_point_displacement:
            l.append({'params': [self.displacements],
                      'lr': training_args.displacement_lr, "name": "displacements"})
        else:
            l.append({'params': self.displacementEncoder.parameters(),
                      'lr': training_args.displacement_encoder_lr, "name": "displacement_encoder"})
        if self.use_point_color:
            l.append({'params': [self.shs_dc],
                      'lr': training_args.shs_lr, "name": "shs"})
            l.append({'params': [self.shs_rest],
                      'lr': training_args.shs_lr / 20.0, "name": "shs"})
        else:
            l.append({'params': self.shEncoder.parameters(),
                      'lr': training_args.sh_encoder_lr, "name": "sh_encoder"})
        return torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def get_points(self):
        if self.use_point_displacement:
            v_displaced = self.v_template + self.displacements
        else:
            v_displaced = self.v_template + \
                          self.displacementEncoder(self.normalized_vertices)
        v_displaced = v_displaced[:, self.verts2uv_idx]
        pass

    def forward(self, body_pose, global_orient, transl, time, is_use_ao=False, **kwargs):
        """
        Caculate the transforms of vertices.

        Args:
            body_pose (torch.Tensor[num_players, J-1, 3]) : The local rotate angles of joints except root joint.
            global_orient (torch.Tensor[num_players, 3]) : The global rotate angle of root joint.
            transl (torch.Tensor[num_players, 3]) : The global translation of root joint.
            time (torch.Tensor[max_freq * 2 + 1]) : Time normalized to 0-1.

        Returns:
            vertices (torch.Tensor[N, 3]) :
            opacity (torch.Tensor[N, 1]) :
            scales (torch.Tensor[N, 3]) :
            rotations (torch.Tensor[N, 4]) :
            shs (torch.Tensor[N, (max_sh_degree + 1) ** 2, 3]) :
            transforms (torch.Tensor[N, 3]) :
        """
        full_body_pose = torch.cat(
            [global_orient[:, None, :], body_pose], dim=1)

        if self.use_point_color:
            shs = torch.cat([self.shs_dc, self.shs_rest], dim=1)
        else:
            shs = self.shEncoder(self.normalized_vertices)

        if self.use_point_displacement:
            v_displaced = self.v_template + self.displacements
        else:
            v_displaced = self.v_template + \
                          self.displacementEncoder(self.normalized_vertices)
        v_displaced = v_displaced[:, self.verts2uv_idx]

        T = lbs(full_body_pose, transl, self.J, self.parents, self.weights)
        T = T[:, self.verts2uv_idx]

        transforms = T[:, :, :3, :].reshape([-1, 3, 4])
        R = transforms[:, :, :3]
        t = transforms[:, :, 3]
        # apply lbs
        verts_posed = torch.matmul(R, v_displaced.reshape(-1, 3).unsqueeze(-1)).squeeze(-1) + t  # N X 3

        from pytorch3d.structures import Meshes
        t_mesh = Meshes(verts=verts_posed.unsqueeze(0), faces=self.faces)
        verts_normal = t_mesh.verts_normals_packed()
        faces_normal = t_mesh.faces_normals_packed()

        return verts_posed.reshape([-1, 3]), torch.sigmoid(self.opacity)[self.verts2uv_idx], \
            torch.exp(self.scales)[self.verts2uv_idx], \
            torch.nn.functional.normalize(self.rotations)[self.verts2uv_idx], \
            shs, T[:, :, :3, :].reshape([-1, 3, 4]), verts_posed, verts_normal

    def lbs(self, body_pose, global_orient, transl):
        batch_size = global_orient.shape[0]
        full_body_pose = torch.cat(
            [global_orient[:, None, :], body_pose], dim=1)

        J = self.J.repeat((batch_size, 1, 1)) * self.model_scale  # repeat J to match batch size
        T = lbs(full_body_pose, transl, J, self.parents, self.weights)
        return T

    def get_joints_from_pose(self, body_pose, global_orient, transl, **kwargs):
        batch_size = global_orient.shape[0]
        T = self.lbs(body_pose, global_orient, transl)
        transforms = T[:, :, :3, :].reshape([batch_size, T.shape[1], 3, 4])
        # transform from canonical to posed space
        R = transforms[:, :, :, :3]
        t = transforms[:, :, :, 3]

        if self.use_point_displacement:
            v_displaced = self.v_template + self.displacements
        else:
            v_displaced = self.v_template * self.model_scale + \
                          self.displacementEncoder(self.normalized_vertices)
        v_displaced = v_displaced.repeat((batch_size, 1, 1))
        vt = torch.matmul(R, v_displaced.unsqueeze(-1)).squeeze(-1) + t
        joints = vertices2joints(self.J_regressor, vt[:, :6890, :])

        # TODO: create independent vertex_joint_selector

        joints = self.body_model.vertex_joint_selector(vt, joints)
        return joints

    def get_2d_pose(self, camera, body_pose, global_orient, transl, smpl2body25=True, **kwargs):
        joints3d = self.get_joints_from_pose(body_pose, global_orient, transl)
        projection_matrices = camera["intrinsic"] @ camera["extrinsic"][:3]
        p = torch.einsum("ij,mnj->mni", projection_matrices[:3, :3], joints3d) + projection_matrices[:3, 3]
        p = p[..., :2] / p[..., 2:3]
        if smpl2body25:
            p = p[:, SMPL_TO_BODY25]
        return p

    def create_uv(self):
        rgbs = self.get_vertex_colors()
        uv_colors = rgbs[self.verts2uv_idx_]
        uv_map_size = 512 // 2
        uv_map = torch.zeros((uv_map_size, uv_map_size, 3), device='cuda')
        for i, (x, y) in enumerate(self.uvs[0]):
            x, y = int(x * uv_map_size), int(y * uv_map_size)
            uv_map[x, y] += uv_colors[i]

        uv_map_ = uv_map.detach().cpu().numpy() * 255.0

        return uv_map_

    def get_vertex_colors(self):
        def sh2rgb(sh):
            C0 = 1 / (np.sqrt(4 * np.pi))
            return sh * C0 + 0.5

        if self.use_point_color:
            shs = torch.cat([self.shs_dc, self.shs_rest], dim=1)
        else:
            shs = self.shEncoder(self.normalized_vertices)

        base_shs = shs[:, 0, :]
        rgb = sh2rgb(base_shs)

        # Initialize vertex colors with zeros
        vertex_colors = rgb
        return vertex_colors
