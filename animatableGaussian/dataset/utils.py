import numpy as np


def kp_to_bbox_param(kp, vis_thresh):
    """
    Finds the bounding box parameters from the 2D keypoints.

    Args:
        kp (Kx3): 2D Keypoints.
        vis_thresh (float): Threshold for visibility.

    Returns:
        [center_x, center_y, scale]
    """
    if kp is None:
        return
    vis = kp[:, 2] > vis_thresh
    if not np.any(vis):
        return
    min_pt = np.min(kp[vis, :2], axis=0)
    max_pt = np.max(kp[vis, :2], axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height < 0.5:
        return
    center = (min_pt + max_pt) / 2.
    scale = 150. / person_height
    return np.append(center, scale)

def get_all_bbox_params(kps, vis_thresh=2):
    """
    Finds bounding box parameters for all keypoints.

    Look for sequences in the middle with no predictions and linearly
    interpolate the bbox params for those

    Args:
        kps (list): List of kps (Kx3) or None.
        vis_thresh (float): Threshold for visibility.

    Returns:
        bbox_params, start_index (incl), end_index (excl)
    """
    # keeps track of how many indices in a row with no prediction
    num_to_interpolate = 0
    start_index = -1
    bbox_params = np.empty(shape=(0, 3), dtype=np.float32)

    for i, kp in enumerate(kps):
        bbox_param = kp_to_bbox_param(kp, vis_thresh=vis_thresh)
        if bbox_param is None:
            num_to_interpolate += 1
            continue

        if start_index == -1:
            # Found the first index with a prediction!
            start_index = i
            num_to_interpolate = 0

        if num_to_interpolate > 0:
            # Linearly interpolate each param.
            previous = bbox_params[-1]
            # This will be 3x(n+2)
            interpolated = np.array(
                [np.linspace(prev, curr, num_to_interpolate + 2)
                 for prev, curr in zip(previous, bbox_param)])
            bbox_params = np.vstack((bbox_params, interpolated.T[1:-1]))
            num_to_interpolate = 0
        bbox_params = np.vstack((bbox_params, bbox_param))

    return bbox_params, start_index, i - num_to_interpolate + 1
