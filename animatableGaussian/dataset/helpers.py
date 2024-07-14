# from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import os

import sys
import json
import torch

sys.path.append('/home/pramish/Desktop/Codes/Animatable-3D-Gaussian/submodules/mediapipe-pose-func')
from pose_predict.predict import predict_pose


#
# CHECKPOINT = os.path.expanduser("~/segment-anything/ckpts/sam_vit_h_4b8939.pth")
# MODEL = "vit_h"
#
# sam = sam_model_registry[MODEL](checkpoint=CHECKPOINT)
# predictor = SamPredictor(sam)
#
#
# def run_sam(img, pts):
#     '''
#
#     Args:
#         img: np.ndarray H X W X 3
#         pts: nK X 3
#
#     Returns:
#
#     '''
#
#     predictor.set_image(img)
#     m = pts[..., -1] > 0.5
#     pts = pts[m]
#     masks, _, _ = predictor.predict(pts[:, :2], np.ones_like(pts[:, 0]))
#     mask = masks.sum(axis=0) > 0
#     return mask
#

def run_rembg(img):
    from rembg.bg import remove

    with torch.no_grad():
        output = remove(img)
        alpha = output[:, :, 3]
        mask = (alpha > 0).astype(int)
    return mask


def run_mediapipe_pose(img):
    result = predict_pose(img)

    # write json output
    keypoints = json.dumps(result['people'])
    keypoints = json.loads(keypoints)
    keypoints = {
        "people": keypoints
    }
    return keypoints
