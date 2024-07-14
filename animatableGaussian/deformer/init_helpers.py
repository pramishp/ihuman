import torch
import numpy as np
import torch.nn.functional as F
from pytorch3d.transforms import (
    matrix_to_quaternion,
)


def s_inv_act(x, min_s_value, max_s_value):
    if isinstance(x, float):
        x = torch.tensor(x).squeeze()
    y = (x - min_s_value) / (max_s_value - min_s_value) + 1e-5
    y = torch.logit(y)
    assert not torch.isnan(
        y
    ).any(), f"{x.min()}, {x.max()}, {y.min()}, {y.max()}"
    return y


def init_qs_on_mesh(
        v_init,
        faces,
        normal,
        scale_init_factor,
        thickness_init_factor,
        max_scale,
        min_scale,
):
    # * Quaternion
    # each column is a basis vector
    # the local frame is z to normal, xy on the disk
    # normal = body_mesh.vertex_normals.copy()
    # v_init = torch.as_tensor(body_mesh.vertices.copy())
    # faces = torch.as_tensor(body_mesh.faces.copy())

    uz = torch.as_tensor(normal, dtype=torch.float32)
    rand_dir = torch.randn_like(uz)
    ux = F.normalize(torch.cross(uz, rand_dir, dim=-1), dim=-1)
    uy = F.normalize(torch.cross(uz, ux, dim=-1), dim=-1)
    frame = torch.stack([ux, uy, uz], dim=-1)  # N,3,3
    ret_q = matrix_to_quaternion(frame)

    # * Scaling
    xy = v_init[faces[:, 1]] - v_init[faces[:, 0]]
    xz = v_init[faces[:, 2]] - v_init[faces[:, 0]]
    area = torch.norm(torch.cross(xy, xz, dim=-1), dim=-1) / 2
    vtx_nn_area = torch.zeros_like(v_init[:, 0])
    for i in range(3):
        vtx_nn_area.scatter_add_(0, faces[:, i], area / 3.0)
    radius = torch.sqrt(vtx_nn_area / np.pi)
    # radius = torch.clamp(radius * scale_init_factor, max=max_scale, min=min_scale)
    # ! 2023.11.22, small eps
    radius = torch.clamp(
        radius * scale_init_factor, max=max_scale - 1e-4, min=min_scale + 1e-4
    )
    thickness = radius * thickness_init_factor
    # ! 2023.11.22, small eps
    thickness = torch.clamp(thickness, max=max_scale - 1e-4, min=min_scale + 1e-4)
    radius_logit = s_inv_act(radius, min_scale, max_scale)
    thickness_logit = s_inv_act(thickness, min_scale, max_scale)
    ret_s = torch.stack([thickness_logit, radius_logit, radius_logit], dim=-1)

    return ret_q, ret_s


def get_init_complex_n_scale_triangles(vertices, faces, surface_triangle_circle_radius,
                                       n_gaussians_per_surface_triangle=1):
    '''

    Args:
        vertices: N X 3
        faces: F X 3

    Returns:

    '''

    n_points = faces.shape[0] * n_gaussians_per_surface_triangle
    faces_verts = vertices[faces]

    # Then, compute initial scales
    scales = (faces_verts - faces_verts[:, [1, 2, 0]]).norm(dim=-1).min(dim=-1)[0] * surface_triangle_circle_radius
    scales = scales.clamp_min(0.0000001).reshape(len(faces_verts), -1, 1).expand(-1, n_gaussians_per_surface_triangle,
                                                                                 2).clone().reshape(-1, 2)
    scales = torch.log(scales)

    # We actually don't learn quaternions here, but complex numbers to encode a 2D rotation in the triangle's plane
    complex_numbers = torch.zeros(n_points, 2)
    complex_numbers[:, 0] = 1.

    return complex_numbers, scales
