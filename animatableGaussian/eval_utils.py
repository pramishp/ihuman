# Some functions are borrowed from https://github.com/akanazawa/human_dynamics/blob/master/src/evaluation/eval_util.py
# Adhere to their licence to use these functions
import pytorch3d.structures
import torch
import numpy as np


def compute_accel(joints):
    """
    Computes acceleration of 3D joints.
    Args:
        joints (Nx25x3).
    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return np.mean(acceleration_normed, axis=1)


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)


def compute_error_verts(pred_verts, target_verts=None, target_theta=None):
    """
    Computes MPJPE over 6890 surface vertices.
    Args:
        verts_gt (Nx6890x3).
        verts_pred (Nx6890x3).
    Returns:
        error_verts (N).
    """

    assert len(pred_verts) == len(target_verts)
    error_per_vert = np.sqrt(np.sum((target_verts - pred_verts) ** 2, axis=2))
    return np.mean(error_per_vert, axis=1)


def p2s(meshes, pcls, reduction='mean'):
    from pytorch3d.structures import Meshes
    from pytorch3d.structures import Pointclouds
    from pytorch3d.loss.point_mesh_distance import point_face_distance

    _DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3

    def point_mesh_face_distance(
            meshes: Meshes,
            pcls: Pointclouds,
            min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Computes the distance between a pointcloud and a mesh within a batch.
        Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
        sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

        `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
            to the closest triangular face in mesh and averages across all points in pcl
        `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
            mesh to the closest point in pcl and averages across all faces in mesh.

        The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
        and then averaged across the batch.

        Args:
            meshes: A Meshes data structure containing N meshes
            pcls: A Pointclouds data structure containing N pointclouds
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.

        Returns:
            loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
                between all `(mesh, pcl)` in a batch averaged across the batch.
        """

        if len(meshes) != len(pcls):
            raise ValueError("meshes and pointclouds must be equal sized batches")
        N = len(meshes)

        # packed representation for pointclouds
        points = pcls.points_packed()  # (P, 3)
        points_first_idx = pcls.cloud_to_packed_first_idx()
        max_points = pcls.num_points_per_cloud().max().item()

        # packed representation for faces
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        tris = verts_packed[faces_packed]  # (T, 3, 3)
        tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
        max_tris = meshes.num_faces_per_mesh().max().item()

        # point to face distance: shape (P,)
        point_to_face = point_face_distance(
            points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
        )
        return point_to_face

    distances = point_mesh_face_distance(meshes, pcls)
    if reduction == "none":
        return distances
    if reduction == 'sum':
        return distances.sum()
    return distances.mean()


def compute_vertex_to_vertex_distance_pytorch3d(source_mesh, target_mesh):
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Pointclouds
    from pytorch3d.loss import chamfer_distance

    # Sample points uniformly from the source and target meshes
    source_pcd = sample_points_from_meshes(source_mesh, num_samples=len(source_mesh.verts_packed()))
    target_pcd = sample_points_from_meshes(target_mesh, num_samples=len(source_mesh.verts_packed()))

    # Convert the sampled points into Pointclouds objects
    source_points = Pointclouds(points=source_pcd)
    target_points = Pointclouds(points=target_pcd)

    # Compute the Chamfer distance between the two point clouds
    # Chamfer distance includes two terms: one for each direction (source to target, target to source)
    # Here, we're interested in the distance from the source to the target, which is the first term returned
    dist_src_to_tgt, d = chamfer_distance(source_points, target_points)

    # The Chamfer distance returns a squared distance, so take the square root to get the actual distances
    # Also, we detach and convert the tensor to a numpy array if needed outside PyTorch
    distances = torch.sqrt(dist_src_to_tgt).detach().cpu().numpy()

    return distances


def compute_vertex_to_vertex_distance_o3d(source_mesh: pytorch3d.structures.Meshes,
                                          target_mesh: pytorch3d.structures.Meshes):
    from pytorch3d.ops import sample_points_from_meshes
    import open3d as o3d

    # Sample points uniformly from the source and target meshes
    source_points = sample_points_from_meshes(source_mesh, num_samples=len(source_mesh.verts_packed()))
    target_points = sample_points_from_meshes(target_mesh, num_samples=len(source_mesh.verts_packed()))

    # Convert the sampled points into Pointclouds objects
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points.detach().cpu().numpy()[0])
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points.detach().cpu().numpy()[0])

    distances_s2t = source_pcd.compute_point_cloud_distance(target_pcd)
    distances_s2t = np.asarray(distances_s2t)
    distances_t2s = target_pcd.compute_point_cloud_distance(source_pcd)
    distances_t2s = np.asarray(distances_t2s)

    mean = ((distances_s2t + distances_t2s) / 2).mean()
    return mean


def compute_centroid_and_scale(vertices):
    centroid = vertices.mean(dim=0)
    scale = (vertices - centroid).norm(dim=1).mean()
    return centroid, scale


# Function to align two meshes based on their centroids and scale
def align_meshes(source_verts, target_verts, adjust_scale=False):
    # Compute centroids and scales
    source_centroid, source_scale = compute_centroid_and_scale(source_verts)
    target_centroid, target_scale = compute_centroid_and_scale(target_verts)

    # Compute scale factor and apply it
    if adjust_scale:
        scale_factor = target_scale / source_scale
        scaled_source_verts = (source_verts - source_centroid) * scale_factor + source_centroid
    else:
        scaled_source_verts = (source_verts - source_centroid) + source_centroid

    # Translate source vertices to target centroid
    translated_source_verts = scaled_source_verts - source_centroid + target_centroid

    return translated_source_verts


def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1 ** 2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # print('X1', X1.shape)

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2)

    # print('var', var1.shape)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # print('R', X1.shape)

    # 5. Recover scale.
    scale = torch.trace(R.mm(K)) / var1
    # print(R.shape, mu1.shape)
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))
    # print(t.shape)

    # 7. Error:
    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat


def align_by_pelvis(joints):
    """
    Assumes joints is 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """

    left_id = 2
    right_id = 3

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    return joints - np.expand_dims(pelvis, axis=0)


def compute_errors(gt3ds, preds):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 14 common joints.
    Inputs:
      - gt3ds: N x 14 x 3
      - preds: N x 14 x 3
    """
    errors, errors_pa = [], []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_pelvis(gt3d)
        pred3d = align_by_pelvis(pred)

        joint_error = np.sqrt(np.sum((gt3d - pred3d) ** 2, axis=1))
        errors.append(np.mean(joint_error))

        # Get PA error.
        pred3d_sym = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
        errors_pa.append(np.mean(pa_error))

    return errors, errors_pa
