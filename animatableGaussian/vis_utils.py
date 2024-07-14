from dataclasses import dataclass

import numpy as np
import cv2
import torch


@dataclass
class OPENPOSE_SKELETON:
    PARTS = [
        (0, 1), (0, 15), (15, 17), (0, 16), (16, 18), (1, 8), (8, 9), (9, 10), (10, 11),
        (11, 22), (22, 23), (11, 24), (8, 12), (12, 13), (13, 14), (14, 21), (14, 19),
        (19, 20), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    ]
    JOINTS = [
        "Nose", "Neck", "(R) Shoulder", "(R) Elbow", "(R) Wrist", "(L) Shoulder", "(L) Elbow",
        "(L) Wrist", "Mid Hip", "(R) Hip", "(R) Knee", "(R) Ankle", "(L) Hip", "(L) Knee",
        "(L) Ankle", "(R) Eye", "(L) Eye", "(R) Ear", "(L) Ear", "(L) B. Toe", "(L) S. Toe",
        "(L) Heel", "(R) B. Toe", "(R) S. Toe", "(R) Heel",
    ]
    COLORS = [
        (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
        (85, 255, 0), (0, 255, 0), (255, 0, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
        (0, 170, 255), (0, 85, 255), (0, 0, 255), (255, 0, 170), (170, 0, 255), (255, 0, 255),
        (85, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 255), (0, 255, 255),
        (0, 255, 255)
    ]


def visualize_pose(image, pose, color=(0, 0, 255)):
    import cv2

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().permute(1, 2, 0).numpy()
        image = image.copy()

    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()

    # Draw circles at the pose coordinates
    for point in pose:
        x, y = point[0], point[1]
        x = int(x)
        y = int(y)
        cv2.circle(image, (x, y), 5, color, -1)  # Draw a circle
    return image


def visualize_pose_3d(joints_3d):
    """

    Args:
        joints_3d: J X 3: np.ndarray

    Returns:

    """
    import open3d as o3d

    if isinstance(joints_3d, torch.Tensor):
        joints_3d = joints_3d.detach().cpu().numpy()

    lines = []
    colors = []
    # Create lines for bones
    for i, (x, y) in enumerate(OPENPOSE_SKELETON.PARTS):
        color = OPENPOSE_SKELETON.COLORS[i]
        lines.append([x, y])
        colors.append(color)

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(joints_3d)

    # Create Open3D line set for bones
    line_set = o3d.geometry.LineSet()
    line_set.points = point_cloud.points
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # line_set.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([point_cloud, line_set])
    return line_set


def resize_image(shape):
    def resize(img, current_frame_id):
        return cv2.resize(img, shape)

    return resize


from aitviewer.headless import HeadlessRenderer
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.point_clouds import PointClouds

viewer = HeadlessRenderer()


def render_model_to_image(verts, camera, imgs, save_path=None, ):
    viewer.reset()
    intrinsic = camera["intrinsic"].detach().cpu().numpy()
    # intrinsic[:2] *= 2
    extrinsic = camera["extrinsic"].detach().cpu().numpy()
    # extrinsic[1:] *= -1
    H = camera["height"]
    W = camera["width"]
    cam = OpenCVCamera(intrinsic, extrinsic[:3], W, H, viewer=viewer)
    viewer.scene.add(cam)

    pc = Billboard.from_camera_and_distance(cam, 15, W, H, imgs, image_process_fn=resize_image((H, W)))

    viewer.scene.add(pc)

    # verts as PC
    # verts[:, :, -1] = verts[:, :, -1] - 1
    pointcloud = PointClouds(verts)
    viewer.scene.add(pointcloud)

    # viewr settings
    viewer.set_temp_camera(cam)
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    viewer.shadows_enabled = False
    if save_path:
        viewer.save_frame(save_path)
    frame = viewer.get_frame()
    return np.asarray(frame)


def render_model(verts, camera):
    viewer.reset()
    intrinsic = camera["intrinsic"].detach().cpu().numpy()
    # intrinsic[:2] *= 2
    extrinsic = camera["extrinsic"].detach().cpu().numpy()
    # extrinsic[1:] *= -1
    H = camera["height"]
    W = camera["width"]
    cam = OpenCVCamera(intrinsic, extrinsic[:3], W, H, viewer=viewer)
    viewer.scene.add(cam)

    # verts as PC
    # verts[:, :, -1] = verts[:, :, -1] - 1
    pointcloud = PointClouds(verts)
    viewer.scene.add(pointcloud)

    # viewr settings
    viewer.set_temp_camera(cam)
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    viewer.shadows_enabled = False

    return viewer.get_frame()


def render_model_front_n_side(verts, camera, side_view_angle=np.pi / 2):
    viewer.reset()
    extrinsic = camera["extrinsic"]

    frame_front = render_model(verts, camera)

    # Calculate center of the model
    center = verts.mean(axis=1)

    # Translate vertices to origin
    verts_centered = verts - center

    # Rotation matrix around the Y-axis
    rotation_matrix = np.array([
        [np.cos(side_view_angle), 0, np.sin(side_view_angle)],
        [0, 1, 0],
        [-np.sin(side_view_angle), 0, np.cos(side_view_angle)]
    ])

    # Apply rotation
    verts_rotated = np.dot(verts_centered, rotation_matrix.T)

    # Translate vertices back
    verts_transformed = verts_rotated + center

    frame_side = render_model(verts_transformed, camera)

    front_side = np.hstack([frame_front, frame_side])
    return np.asarray(front_side)


def create_side_by_side_images(gt_images, pred_images):
    # Check that gt_images and pred_images have the same shape
    assert gt_images.shape == pred_images.shape

    # Initialize an empty tensor for side-by-side comparisons
    batch_size, c, h, w = gt_images.shape
    comparisons = torch.empty(batch_size * 2, c, h, w)

    # Interleave GT and pred images
    comparisons[0::2] = gt_images.cpu()
    comparisons[1::2] = pred_images.detach().cpu()
    # Use make_grid to create a grid of the interleaved images
    return make_grid(comparisons, nrow=2)


def make_grid(images, nrow=2):
    import torchvision

    grid = torchvision.utils.make_grid(images, nrow=nrow)

    npimg = grid.numpy()
    return np.transpose(npimg, (1, 2, 0))


def create_pose_comparison_image(gt_images, rasterized_images, gt_poses, model_poses, optimized_poses=None):
    batch_size = len(gt_images)
    type2color = {
        'gt': (0, 255.0, 0),
        'model': (0, 0, 255.0),
        'optimized': (255.0, 0, 0)
    }
    gt_pose_images = []
    model_pose_images = []

    for i in range(len(gt_images)):
        gt_pose = gt_poses[i]
        pose_mask = gt_pose[..., 2] > 0.2
        model_pose = model_poses[i][pose_mask]
        gt_pose = gt_pose[pose_mask]

        rgb_gt = gt_images[i] * 255
        rgb_rasterized = rasterized_images[i] * 255

        gt_pose_image = visualize_pose(rgb_gt, gt_pose, color=type2color['gt'])

        model_pose_img = visualize_pose(rgb_rasterized,
                                        model_pose, color=type2color['model'])
        # visualize GT over model pose image
        model_pose_img = visualize_pose(model_pose_img.copy(),
                                        gt_pose, color=type2color['gt'])

        if optimized_poses is not None:
            optimized_pose = optimized_poses[i][pose_mask]
            model_pose_img = visualize_pose(model_pose_img.copy(), optimized_pose, color=type2color['optimized'])

        model_pose_images.append(torch.from_numpy(model_pose_img))
        gt_pose_images.append(torch.from_numpy(gt_pose_image))

    model_pose_images = torch.stack(model_pose_images).permute(0, 3, 1, 2)
    gt_pose_images = torch.stack(gt_pose_images).permute(0, 3, 1, 2)

    grid = create_side_by_side_images(gt_pose_images, model_pose_images)
    return grid


def novel_view_as_video(model, camera_params, model_param, output_video_path):
    from PIL import Image
    # Initialize video writer
    image_width, image_height = camera_params["image_width"], camera_params["image_height"]
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(output_video_path, fourcc, 10, (image_width, image_height))

    global_orient = model_param['global_orient']

    for angle in np.arange(0, 360, 5):
        rad = np.pi * angle / 180
        new_global_orient = global_orient.clone()
        new_global_orient[:, 2] = rad

        # Assume `self` is an instance of a class that contains the method being used to generate images
        rgb, _, _, _, _, _, _ = model(camera_params, model_param, 0, train=False)

        image = rgb
        img = (255.0 * image.permute(1, 2, 0)).data.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)

        # Convert PIL image to an OpenCV image
        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # Write frame to video
        video.write(open_cv_image)

    # Release the video writer
    video.release()

    print(f"Video saved as {output_video_path}")
