import numpy as np
import torch
from dataclasses import dataclass
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
import numpy as np


def projection(pred_joints, pred_camera, im_size=224):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * (5000 / 2) / (im_size * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(
                                                   pred_joints.device),
                                               translation=torch.zeros_like(pred_cam_t),
                                               focal_length=5000. / 2,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (im_size / 2)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


@dataclass
class Camera:
    """
    Attributes:
        image_height (int) : Height of the rendered image.
        image_width (int) : Width of the rendered image.
        tanfovx (float) : image_width / (2 * focal_x).
        tanfovy (float) : image_height / (2 * focal_y).
        bg (torch.Tensor[3, image_height, image_width]) : The backgroud image of the rendered image.
        scale_modifier (float) : Global scaling of 3D gaussians.
        viewmatrix (torch.Tensor[4, 4]) : Viewmatrix (column main order, the transpose of the numpy matrix).
        projmatrix (torch.Tensor[4, 4]) : The product of the projmatrix and viewmatrix (column main order, the transpose of the numpy matrix).
        campos (torch.Tensor[1, 3]) : The world position of the camera center.
    """
    image_height: int = None
    image_width: int = None
    tanfovx: float = None
    tanfovy: float = None
    bg: torch.Tensor = None
    scale_modifier: float = None
    viewmatrix: torch.Tensor = None
    projmatrix: torch.Tensor = None
    campos: torch.Tensor = None


@dataclass
class ModelParam:
    """
    Attributes:
        body_pose (torch.Tensor[num_players, J-1, 3]) : The local rotate angles of joints except root joint.
        global_orient (torch.Tensor[num_players, 3]) : The global rotate angle of root joint.
        transl (torch.Tensor[num_players, 3]) : The global translation of root joint.
    """
    body_pose: torch.Tensor = None
    global_orient: torch.Tensor = None
    transl: torch.Tensor = None


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                              float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# def visualize_3d(vertices):
#     import plotly.graph_objects as go
#
#     # Separate coordinates into x, y, z lists
#     x_coords = vertices[:, 0].detach().cpu()
#     y_coords = vertices[:, 1].detach().cpu()
#     z_coords = vertices[:, 2].detach().cpu()
#
#     # Create a trace for the vertices
#     trace = go.Scatter3d(
#         x=x_coords,
#         y=y_coords,
#         z=z_coords,
#         mode='markers',  # Show markers and text (vertex numbers)
#         marker=dict(size=8, color='blue'),
#         # text=[f'Vertex {i + 1}' for i in range(len(vertices))],
#         # textposition='top center'
#     )
#
#     # Create the layout
#     layout = go.Layout(
#         scene=dict(
#             xaxis=dict(title='X Axis'),
#             yaxis=dict(title='Y Axis'),
#             zaxis=dict(title='Z Axis')
#         ),
#         margin=dict(l=0, r=0, b=0, t=0)
#     )
#
#     # Create the figure
#     fig = go.Figure(data=[trace], layout=layout)
#
#     # Show the plot
#     fig.show()


def visualize_3d(verts):
    import open3d as o3d

    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()

    # Create an Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(verts)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])


def visualize_3d_mesh(verts, faces):
    import open3d as o3d

    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()

    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    # Create an Open3D point cloud
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([mesh])


def save_ply(verts, path, format='pc'):
    import open3d as o3d

    verts = verts.detach().cpu()

    if format == "pc":
        # Create an Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(verts)
        o3d.io.write_point_cloud(path, point_cloud)
    else:
        raise Exception("Not implemented error")


def save_mesh(verts, faces, path, format='obj'):
    import open3d as o3d

    verts = verts.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()

    if format == "obj":
        # Create an Open3D point cloud
        point_cloud = o3d.geometry.TriangleMesh()
        point_cloud.vertices = o3d.utility.Vector3dVector(verts)
        point_cloud.triangles = o3d.utility.Vector3iVector(faces)
        o3d.io.write_triangle_mesh(path, point_cloud)
    else:
        raise Exception("Not implemented error")


class GMoF(torch.nn.Module):
    def __init__(self, rho=1):
        super().__init__()
        self.rho = rho

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist


# ply file implementation with all components(point, scolor, opacity, rotation , scaling)

class SavePly:
    # TODO : Use transforms as well to ensure the model is in posed space
    def __init__(self, verts, opacity, scales, rotations, shs):
        self._xyz = verts
        # NOTE :this makes the _features a two dimensional array. Do this without making it a 2 dim array
        self._features_dc = shs[:, 0:1, :]
        self._features_rest = shs[:, 1:, :]
        self._opacity = opacity
        self._scaling = scales
        self._rotation = rotations

        pass

    def list_of_attributes(self):
        '''
            creates a list of attributes that are to be saved
            points and their normals
            features_dc : the base color, SHO
            features_rest: rest of the spherical harmonics
            opacity
            scaling
            rotation
        '''
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append(f'f_dc_{i}')
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')
        return l

    def save_ply(self, path):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


def save_image(img, path):
    from PIL import Image

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    factor = 1  # not normalized image
    if img.max() <= 1:
        factor = 255.  # normalized image case
    img = (factor * img).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)


def many2one_mapper(one2many):
    # Initialize uv2vert with None or any placeholder to indicate unmapped UVs
    many2one = {}

    # Track used vertices to ensure one-to-one mapping in uv2vert
    used_one_set = set()

    for uv_idx, vert_idx in enumerate(one2many):
        # Check if the vertex has already been mapped to a UV; if not, map the current UV to this vertex
        if vert_idx not in used_one_set and many2one.get(uv_idx, None) is None:
            many2one[uv_idx] = vert_idx
            used_one_set.add(vert_idx)

    uv2vert = many2one
    sorted_uv_indices = list(map(lambda item: item[0], sorted(uv2vert.items(), key=lambda item: item[1])))

    return np.asarray(sorted_uv_indices)


def normalize(v):
    """Normalize a tensor of vectors."""
    return v / v.norm(dim=-1, keepdim=True)

def construct_rotation_matrix(axis, angle):
    """Construct a rotation matrix from an axis and an angle using Rodrigues' rotation formula."""
    axis = normalize(axis)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    C = 1 - cos

    xs, ys, zs = axis[..., 0], axis[..., 1], axis[..., 2]
    xC, yC, zC = C * xs, C * ys, C * zs
    xSin, ySin, zSin = sin * xs, sin * ys, sin * zs

    # Construct the rotation matrix
    R = torch.stack([
        torch.stack([cos + xC * xs, xC * ys - zSin, xC * zs + ySin], dim=-1),
        torch.stack([yC * xs + zSin, cos + yC * ys, yC * zs - xSin], dim=-1),
        torch.stack([zC * xs - ySin, zC * ys + xSin, cos + zC * zs], dim=-1)
    ], dim=-2)

    return R
def compute_rotation_matrix(canonical_normals, transformed_normals):
    """Compute the rotation matrices that align canonical_normals with transformed_normals."""
    # Normalize the input normals to ensure they are unit vectors
    canonical_normals = normalize(canonical_normals)
    transformed_normals = normalize(transformed_normals)

    # Compute the rotation axis as the cross product of the two normals
    axis = torch.cross(canonical_normals, transformed_normals, dim=1)

    # Compute the cosine of the rotation angle using the dot product
    cos_angle = torch.sum(canonical_normals * transformed_normals, dim=1)

    # Ensure the cosine value is in the valid range [-1, 1] to avoid numerical issues
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    # Compute the rotation angle using arccos
    angle = torch.acos(cos_angle)

    # Construct the rotation matrix using Rodrigues' rotation formula
    rotation_matrices = construct_rotation_matrix(axis, angle)

    return rotation_matrices
