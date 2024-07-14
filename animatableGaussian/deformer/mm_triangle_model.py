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
                 use_point_color=False, use_point_displacement=False,
                 model_size="large"):
        super().__init__()

        self.num_players = num_players
        self.use_point_displacement = use_point_displacement
        self.use_point_color = use_point_color
        self.n_gaussians_per_surface_triangle = 1

        # smpl_path = os.path.join(
        #     model_path, 'SMPL_{}'.format(gender.upper()))
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

        uv = np.loadtxt(os.path.join(smpl_path, 'uv_coords.txt')).astype(np.float32)
        uv = torch.from_numpy(uv).unsqueeze(0)

        # self.face2uv_idx = np.loadtxt(os.path.join(smpl_path, 'uv_map2face.txt')).astype(int)
        self.face2uv_idx = np.arange(0, faces.shape[0])
        # self.verts2uv_idx = self.face2uv_idx

        self.register_buffer('uvs', uv.repeat([self.num_players, 1, 1]))

        self.register_buffer('v_template', torch.Tensor(
            v_template)[None, ...].repeat(
            [self.num_players, 1, 1]))
        dist2 = torch.clamp_min(
            distCUDA2(self.v_template[0].cuda()), 0.0000001)[..., None].repeat([num_repeat, 3])
        dist2 /= num_repeat

        # self.v_template = self.v_template.repeat([1, num_repeat, 1])
        ## pytorch 3d mesh
        from pytorch3d.structures import Meshes

        self.mesh = Meshes(verts=self.v_template, faces=self.faces).to('cuda')
        self.mesh._compute_vertex_normals()
        self.register_buffer("vertex_normals", self.mesh.verts_normals_packed())  # NX3

        self.weights = nn.Parameter(
            torch.Tensor(weights[self.mesh2smpl_idx]).repeat([num_repeat, 1]))
        self.parents = kintree_table[0].astype(np.int64)
        self.parents[0] = -1

        self.J = nn.Parameter(torch.Tensor(
            J)[None, ...].repeat([self.num_players, 1, 1]))

        minmax = [self.v_template[0].min(
            dim=0).values * 1.05, self.v_template[0].max(dim=0).values * 1.05]

        normalized_vertices = (self.v_template - minmax[0]) / (minmax[1] - minmax[0])
        self.register_buffer('normalized_vertices',
                             normalized_vertices)

        normalized_faces = normalized_vertices[0][self.faces].sum(dim=-2)
        self.register_buffer('normalized_faces', normalized_faces)

        if use_point_displacement:
            self.displacements = nn.Parameter(
                torch.zeros_like(self.v_template))
        else:
            self.displacementEncoder = DisplacementEncoder(
                encoder="hash", num_players=num_players)

        f = self.faces.shape[1]
        self.n_points = f * self.n_gaussians_per_surface_triangle

        if use_point_color:
            self.shs_dc = nn.Parameter(torch.zeros(
                [self.n_points, 1, 3]))
            self.shs_rest = nn.Parameter(torch.zeros(
                [self.n_points, (max_sh_degree + 1) ** 2 - 1, 3]))
        else:
            self.shEncoder = SHEncoder(max_sh_degree=max_sh_degree,
                                       encoder="hash", num_players=num_players)
        # self.opacity = nn.Parameter(inverse_sigmoid(
        #     0.2 * torch.ones((n, 1), dtype=torch.float)))
        self.opacity = nn.Parameter(inverse_sigmoid(torch.ones((f, 1)) * 0.99))

        # triangular initializations of gaussian params

        if self.n_gaussians_per_surface_triangle == 1:
            self.surface_triangle_circle_radius = 1. / 2. / np.sqrt(3.)
            self.surface_triangle_bary_coords = torch.tensor(
                [[1 / 3, 1 / 3, 1 / 3]],
                dtype=torch.float32,
                device='cuda',
            )[..., None]

        if self.n_gaussians_per_surface_triangle == 3:
            self.surface_triangle_circle_radius = 1. / 2. / (np.sqrt(3.) + 1.)
            self.surface_triangle_bary_coords = torch.tensor(
                [[1 / 2, 1 / 4, 1 / 4],
                 [1 / 4, 1 / 2, 1 / 4],
                 [1 / 4, 1 / 4, 1 / 2]],
                dtype=torch.float32,
                device='cuda',
            )[..., None]

        self.surface_mesh_thickness = 1e-3

        # scales and rotation
        from animatableGaussian.deformer.init_helpers import get_init_complex_n_scale_triangles

        complex_quat, scaling = get_init_complex_n_scale_triangles(self.mesh.verts_packed(), self.mesh.faces_packed(),
                                                                   self.surface_triangle_circle_radius,
                                                                   n_gaussians_per_surface_triangle=self.n_gaussians_per_surface_triangle)

        self.scales = nn.Parameter(scaling)

        self.rotations = nn.Parameter(complex_quat)

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
                      'lr': training_args.shs_lr, "name": "shs_rest"})
            l.append({'params': [self.shs_rest],
                      'lr': training_args.shs_lr / 20.0, "name": "shs_dc"})
        else:
            l.append({'params': self.shEncoder.parameters(),
                      'lr': training_args.sh_encoder_lr, "name": "sh_encoder"})
        return torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def get_points(self, v_displaced_posed):
        # First gather vertices of all triangles
        faces_verts = v_displaced_posed[self.faces[0]]  # n_faces, 3, n_coords

        # Then compute the points using barycenter coordinates in the surface triangles
        points = faces_verts[:, None] * self.surface_triangle_bary_coords[
            None]  # n_faces, n_gaussians_per_face, 3, n_coords
        points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords

        return points.reshape(1, self.n_points, 3)

    def get_quaternions(self, v_displaced_posed, posed_mesh):
        n_surface_mesh_faces = self.faces.shape[1]
        # n_points = n_surface_mesh_faces * self.n_gaussians_per_surface_triangle

        from pytorch3d.transforms import matrix_to_quaternion

        R_0 = torch.nn.functional.normalize(posed_mesh.faces_normals_list()[0], dim=-1)

        # We use the first side of every triangle as the second base axis
        faces_verts = v_displaced_posed[self.mesh.faces_packed()]
        base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)

        # We use the cross product for the last base axis
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))

        # We now apply the learned 2D rotation to the base quaternion
        complex_numbers = torch.nn.functional.normalize(self.rotations, dim=-1).view(n_surface_mesh_faces,
                                                                                     self.n_gaussians_per_surface_triangle,
                                                                                     2)
        R_1 = complex_numbers[..., 0:1] * base_R_1[:, None] + complex_numbers[..., 1:2] * base_R_2[:, None]
        R_2 = -complex_numbers[..., 1:2] * base_R_1[:, None] + complex_numbers[..., 0:1] * base_R_2[:, None]

        # We concatenate the three vectors to get the rotation matrix
        R = torch.cat([R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
                       R_1[..., None],
                       R_2[..., None]],
                      dim=-1).view(-1, 3, 3)
        quaternions = matrix_to_quaternion(R)

        return torch.nn.functional.normalize(quaternions, dim=-1)

    def get_scales(self):
        plane_scales = torch.exp(self.scales)
        scales = torch.cat([
            self.surface_mesh_thickness * torch.ones(len(self.scales), 1, device=self.scales.device),
            plane_scales,
        ], dim=-1)
        return scales

    def forward(self, body_pose, global_orient, transl, time=0, **kwargs):
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
            shs = self.shEncoder(self.normalized_faces)
            # shs = self.shEncoder(self.uvs)

        if self.use_point_displacement:
            v_displaced = self.v_template + self.displacements
        else:
            v_displaced = self.v_template + \
                          self.displacementEncoder(self.normalized_vertices)
        # v_displaced = v_displaced  # B X N X 3

        # canonical to posed space
        T = lbs(full_body_pose, transl, self.J, self.parents, self.weights)
        # T = T[:, self.verts2uv_idx]

        transforms = T[:, :, :3, :].reshape([-1, 3, 4])
        R = transforms[:, :, :3]
        t = transforms[:, :, 3]

        # posed space verts
        verts_posed = torch.matmul(R, v_displaced.reshape(-1, 3).unsqueeze(-1)).squeeze(-1) + t  # N X 3

        from pytorch3d.structures import Meshes
        posed_mesh = Meshes(verts=verts_posed.unsqueeze(0), faces=self.faces)
        faces_normals = posed_mesh.faces_normals_packed()[self.face2uv_idx]

        f_displaced = self.get_points(verts_posed)
        f_displaced = f_displaced[:, self.face2uv_idx]
        quaternions = self.get_quaternions(verts_posed, posed_mesh)
        quaternions = quaternions[self.face2uv_idx]
        scales = self.get_scales()
        scales = scales[self.face2uv_idx]

        T = torch.zeros((1, f_displaced.shape[1], 4, 4), device=scales.device)
        T[:, :, :3, :3] = torch.eye(3)

        # get rotation for each faces
        # from animatableGaussian.utils import compute_rotation_matrix
        # base_components = shs[:, :1, :]  # Shape [N, 1, 3], the base or DC components
        # higher_order_components = shs[:, 1:, :]  # Shape [N, 8, 3], the higher-order components
        # f_rot = compute_rotation_matrix(self.mesh.faces_normals_packed(), faces_normals)
        # rotated_higher_order_components = torch.einsum('nij,njk->nik', higher_order_components, f_rot)
        # rotated_shs = torch.cat((base_components, rotated_higher_order_components), dim=1)  # Shape [N, 9, 3]

        return f_displaced.reshape(-1, 3), torch.sigmoid(self.opacity)[self.face2uv_idx], \
            scales, \
            quaternions, \
            shs, T[:, :, :3, :].reshape([-1, 3, 4]), verts_posed, faces_normals

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

    def get_vertex_colors(self):

        def sh2rgb(sh):
            C0 = 1 / (np.sqrt(4 * np.pi))
            return sh * C0 + 0.5

        if self.use_point_color:
            shs = torch.cat([self.shs_dc, self.shs_rest], dim=1)
        else:
            shs = self.shEncoder(self.normalized_faces)

        base_shs = shs[:, 0, :]
        rgb = sh2rgb(base_shs)
        # Sample face colors
        face_colors = rgb

        # Initialize vertex colors with zeros
        vertex_colors = torch.zeros_like(self.v_template[0])

        # Count the number of faces each vertex is shared with
        vertex_counts = torch.zeros_like(self.v_template[0][:, 0], dtype=torch.float32)
        for i, face in enumerate(self.faces[0]):
            vertex_colors[face] += face_colors[i]  # Accumulate face colors to vertices
            vertex_counts[face] += 1  # Increment the count of faces for each vertex

        # Divide accumulated colors by the number of faces each vertex is shared with
        vertex_colors /= vertex_counts.unsqueeze(-1).clamp(min=1)

        return vertex_colors
