import logging

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import glob
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from animatableGaussian.dataset.utils import get_all_bbox_params
from animatableGaussian.utils import Camera, ModelParam


def time_encoding(t, dtype, max_freq=4):
    time_enc = torch.empty(max_freq * 2 + 1, dtype=dtype)

    for i in range(max_freq):
        time_enc[2 * i] = np.sin(2 ** i * torch.pi * t)
        time_enc[2 * i + 1] = np.cos(2 ** i * torch.pi * t)
    time_enc[max_freq * 2] = t
    return time_enc


def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    data = {
        "body_pose": torch.from_numpy(smpl_params["body_pose"].astype(np.float32)),
        "global_orient": torch.from_numpy(smpl_params["global_orient"].astype(np.float32)),
        "transl": torch.from_numpy(smpl_params["transl"].astype(np.float32)),
        'betas': torch.from_numpy(smpl_params["betas"].astype(np.float32))
    }
    if 'v_personal' in smpl_params.keys():
        data['v_personal'] = torch.from_numpy(smpl_params['v_personal'].astype(np.float32))

    return data


def focal2tanfov(focal, pixels):
    return pixels / (2 * focal)


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def getProjectionMatrixShift(tanHalfFovY, tanHalfFovX, focal_x, focal_y, cx, cy, width, height, znear=0.01, zfar=100.0):
    # the origin at center of image plane
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # shift the frame window due to the non-zero principle point offsets
    offset_x = cx - (width / 2)
    offset_x = (offset_x / focal_x) * znear
    offset_y = cy - (height / 2)
    offset_y = (offset_y / focal_y) * znear

    top = top + offset_y
    left = left + offset_x
    right = right + offset_x
    bottom = bottom + offset_y

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrix(tanHalfFovY, tanHalfFovX, znear=0.01, zfar=100.0):
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getCamIntrinsicAndExtrinsic(intrinsic, opt):
    focal_length_x = intrinsic["focal_length_x"]
    focal_length_y = intrinsic["focal_length_y"]
    camera_center_x = intrinsic["camera_center_x"]
    camera_center_y = intrinsic["camera_center_y"]

    intr = np.array([[focal_length_x, 0, camera_center_x],
                     [0, focal_length_y, camera_center_y],
                     [0, 0, 1]]).astype(np.float64)

    intr[:2] /= opt.downscale

    extrinsic = torch.eye(4)
    # extrinsic[-1, -1] = 0
    return torch.from_numpy(intr).float(), extrinsic


def getCamPara(camera, opt):
    intr = camera["intrinsic"]
    # intr[0, 2] = camera['width'] // 2
    # intr[1, 2] = camera['height'] // 2
    # # intr = np.asarray([])
    # intr[:2] /= opt.downscale
    c2w = np.linalg.inv(camera["extrinsic"])
    height = int(camera["height"] / opt.downscale)
    width = int(camera["width"] / opt.downscale)
    focal_length_x = intr[0, 0]
    focal_length_y = intr[1, 1]

    tanFovY = focal2tanfov(focal_length_y, height)
    tanFovX = focal2tanfov(focal_length_x, width)
    camera_params = Camera()
    # projmatrix = getProjectionMatrix(tanFovY, tanFovX)
    projmatrix = getProjectionMatrixShift(tanFovY, tanFovX, focal_x=focal_length_x, focal_y=focal_length_y,
                                          cx=intr[0, 2], cy=intr[1, 2], width=width, height=height)
    viewmatrix = torch.Tensor(c2w)

    camera_params.image_height = height
    camera_params.image_width = width
    camera_params.tanfovx = tanFovX
    camera_params.tanfovy = tanFovY
    camera_params.bg = torch.ones([3, height, width])
    camera_params.scale_modifier = 1.0
    camera_params.viewmatrix = viewmatrix.T
    camera_params.projmatrix = (projmatrix @ viewmatrix).T
    camera_params.campos = torch.inverse(camera_params.viewmatrix)[3, :3]
    return camera_params


def read_mask(msk_path):
    msk = Image.open(msk_path)
    msk = np.array(msk).astype(np.float32)
    normalized_mask = msk / msk.max()
    msk = torch.from_numpy(normalized_mask)[None, None, ...]
    return msk


def read_imgs(dataroot, opt, resolution):
    start = opt.start
    end = opt.end + 1
    skip = opt.get("skip", 1)
    img_lists = sorted(
        glob.glob(f"{dataroot}/images/*.png"))[start:end:skip]
    if len(img_lists) == 0:
        img_lists = sorted(
            glob.glob(f"{dataroot}/images/*.jpg"))[start:end:skip]
    msk_lists = sorted(
        glob.glob(f"{dataroot}/masks/*.png"))[start:end:skip]
    if len(msk_lists) == 0:
        msk_lists = sorted(
            glob.glob(f"{dataroot}/masks/*.jpg"))[start:end:skip]
    if len(msk_lists) == 0:
        msk_lists = sorted(
            glob.glob(f"{dataroot}/masks/*.npy"))[start:end:skip]

    frame_count = len(img_lists)
    imgs = []
    masks = []
    for index in tqdm(range(frame_count)):
        image_path = img_lists[index]
        img = Image.open(image_path)
        img = PILtoTorch(img, resolution)

        msk_path = msk_lists[index]
        if ".npy" in msk_lists[0]:
            msk = torch.from_numpy(np.load(msk_path).astype(np.float32))[
                None, None, ...]
        else:

            msk = read_mask(msk_path)

        msk = F.interpolate(msk, scale_factor=1 / opt.downscale, mode='bilinear')
        masked_img = img[:3, ...] * msk[0] + 1 - msk[0]
        imgs.append(masked_img)
        masks.append(msk)
    return imgs, masks


def read_normals(dataroot, opt, resolution):
    start = opt.start
    end = opt.end + 1
    skip = opt.get("skip", 1)
    img_lists = sorted(
        glob.glob(f"{dataroot}/normals/*.png"))[start:end:skip]
    if len(img_lists) == 0:
        img_lists = sorted(
            glob.glob(f"{dataroot}/normals/*.jpg"))[start:end:skip]
    msk_lists = sorted(
        glob.glob(f"{dataroot}/masks/*.png"))[start:end:skip]
    if len(msk_lists) == 0:
        msk_lists = sorted(
            glob.glob(f"{dataroot}/masks/*.jpg"))[start:end:skip]
    if len(msk_lists) == 0:
        msk_lists = sorted(
            glob.glob(f"{dataroot}/masks/*.npy"))[start:end:skip]

    frame_count = len(img_lists)
    imgs = []
    masks = []
    for index in tqdm(range(frame_count)):
        image_path = img_lists[index]
        img = Image.open(image_path)
        img = PILtoTorch(img, resolution)

        msk_path = msk_lists[index]
        if ".npy" in msk_lists[0]:
            msk = torch.from_numpy(np.load(msk_path).astype(np.float32))[
                None, None, ...]
        else:

            msk = read_mask(msk_path)

        msk = F.interpolate(msk, scale_factor=1 / opt.downscale, mode='bilinear')
        masked_img = img[:3, ...] * msk[0] + 1 - msk[0]
        imgs.append(masked_img)
        masks.append(msk)
    return imgs


def read_depth(path):
    import cv2
    # Read the depth map image using OpenCV
    depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    depth_image = depth_image / 1000  # convert to meters from mm
    # Convert the depth image back to the original depth values (assuming it was scaled to 0-255 range)
    depth_map = depth_image.astype(np.float32)  # Convert back to the original depth range
    depth_map = torch.from_numpy(depth_map)[None, None, ...]
    return depth_map


def read_depths(dataroot, opt, resolution):
    # TODO: read depths: modify this func
    start = opt.start
    end = opt.end + 1
    skip = opt.get("skip", 1)
    img_lists = sorted(
        glob.glob(f"{dataroot}/depths/*.png"))[start:end:skip]
    msk_lists = sorted(
        glob.glob(f"{dataroot}/masks/*.png"))[start:end:skip]
    frame_count = len(img_lists)
    depths = []

    for index in tqdm(range(frame_count)):
        image_path = img_lists[index]
        depth = read_depth(image_path)
        depth = F.interpolate(depth, scale_factor=1 / opt.downscale,
                              mode='bilinear')
        msk_path = msk_lists[index]
        msk = read_mask(msk_path)
        msk = F.interpolate(msk, scale_factor=1 / opt.downscale, mode='bilinear')
        depth = depth[0] * msk[0]
        depths.append(depth)
    return depths


def load_pose(dataroot, opt, split, fname="poses.npz"):
    start = opt.start
    end = opt.end + 1
    skip = opt.get("skip", 1)
    skip_pose_base = opt.get('skip_pose_base', 4)
    # if os.path.exists(os.path.join(dataroot, f"poses/fixed_transl/anim_nerf_{split}.npz")):
    #     cached_path = os.path.join(dataroot, f"poses/fixed_transl/anim_nerf_{split}.npz")
    # elif os.path.exists(os.path.join(dataroot, f"poses/fixed_transl/{split}.npz")):
    #     cached_path = os.path.join(dataroot, f"poses/fixed_transl/{split}.npz")
    # else:
    #     cached_path = None
    if os.path.exists(os.path.join(dataroot, f"poses/anim_nerf_{split}.npz")):
        cached_path = os.path.join(dataroot, f"poses/anim_nerf_{split}.npz")
    elif os.path.exists(os.path.join(dataroot, f"poses/{split}.npz")):
        cached_path = os.path.join(dataroot, f"poses/{split}.npz")
    else:
        cached_path = None

    if cached_path and os.path.exists(cached_path):
        print(f"[{split}] Loading from", cached_path)
        smpl_params = load_smpl_param(cached_path)
        for k, v in smpl_params.items():
            if k != "betas" and k != "v_personal":
                smpl_params[k] = v[0:v.shape[0]:skip // skip_pose_base]
        # modify transl to support 540,540 camera center
        # tx, ty = 540-511.78, 567.12-540
        # smpl_params['transl'][:, 0] -= tx / 1200
        # smpl_params['transl'][:, 1] += ty / 1200
    else:
        print(f"[{split}] No optimized smpl found.")
        smpl_params = load_smpl_param(os.path.join(dataroot, fname))
        for k, v in smpl_params.items():
            if k != "betas" and k != "v_personal":
                smpl_params[k] = v[start:end:skip]
    return smpl_params


def load_2d_poses(dataroot, opt):
    start = opt.start
    end = opt.end + 1
    skip = opt.get("skip", 1)
    keypoints = np.load(os.path.join(dataroot, f"keypoints.npy"))[start:end:skip]
    keypoints = torch.from_numpy(keypoints.astype(np.float32))
    return keypoints


class MMDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, max_freq, split, opt, camera, gender):
        self.opt = opt
        self.split = split
        self.max_freq = max_freq
        self.gender = gender
        self.camIntrinsic, self.camExtrinsic = getCamIntrinsicAndExtrinsic(camera, opt)
        camera = {'intrinsic': self.camIntrinsic, 'extrinsic': self.camExtrinsic,
                  'width': camera['width'],
                  'height': camera['height']}
        self.camera_params = getCamPara(camera, opt)
        self.imgs, self.masks = read_imgs(
            dataroot, opt,
            (self.camera_params.image_width, self.camera_params.image_height),
        )

        self.normals = read_normals(dataroot, opt, (self.camera_params.image_width, self.camera_params.image_height))

        self.smpl_params = load_pose(dataroot, opt, split, fname="poses.npz")
        # self.poses_2d = load_2d_poses(dataroot, opt)

        start = opt.start
        end = opt.end + 1
        skip = opt.get("skip", 1)
        self.num_img_seq = 20  # TODO: config in opt
        self.img_lists = sorted(
            glob.glob(f"{dataroot}/images/*.png"))[start:end:skip]
        if len(self.img_lists) == 0:
            self.img_lists = sorted(
                glob.glob(f"{dataroot}/images/*.jpg"))[start:end:skip]

    def __len__(self):
        return len(self.imgs)

    def get_SMPL_params(self):
        # set true pose only to the first frame
        batch_size = self.smpl_params['body_pose'].shape[0]
        params = {}
        for k, v in self.smpl_params.items():
            if k == "betas":
                if len(v.shape) == 1:
                    v = v.unsqueeze(0)
            params[k] = v

        return params

    def __getitem__(self, index):
        """
        Returns:
            data["camera_params"] (vars(Camera)) : Input dict for gaussian rasterizer.
            data["model_param"] (vars(ModelParam)) : Input dict for a deformer model.
            data["gt"] (torch.Tensor[3, h, w]) : Ground truth image.
            data["time"] (torch.Tensor[max_freq * 2 + 1,]) : Time normalized to 0-1.
        """
        t = index / self.__len__()
        #
        smpl_param = {}

        try:
            smpl_param["global_orient"] = self.smpl_params["global_orient"][index]
            smpl_param["body_pose"] = self.smpl_params["body_pose"][index].reshape([-1, 3])
            smpl_param["transl"] = self.smpl_params["transl"][index]
        except:
            print()

        ## use GT model params to get 3d pose and vertices
        # import smplx
        #
        # body_model = smplx.SMPL("../../data/smpl/models/", gender=self.gender).to('cpu')
        # params = {"body_pose": self.smpl_params_gt["body_pose"],
        #           "global_orient": self.smpl_params_gt["global_orient"],
        #           "transl": self.smpl_params_gt["transl"],
        #           'betas': self.smpl_params_gt["betas"].reshape(1, 10),
        #           }
        # if 'v_personal' in self.smpl_params_gt:
        #     params['v_personal'] = self.smpl_params_gt['v_personal'].reshape(6890, 3)
        #
        # smpl_out = body_model(**params, return_verts=True)
        #
        # vertices = smpl_out['vertices'][index]
        # if 'v_perosnal' in self.smpl_params_gt:
        #     vertices += self.smpl_params_gt['v_personal']
        # joints_3d = smpl_out['joints'][index]
        #
        # height = ((vertices[412] - (vertices[3458] + vertices[6858]) / 2).square()).sum().sqrt()
        #
        # # pose downscale
        # pose_2d = self.poses_2d[index].clone()
        # pose_2d[:, :2] = pose_2d[:, :2] / self.opt.downscale

        data = {"camera_params": vars(self.camera_params),
                "model_param": smpl_param,
                "gt": self.imgs[index],
                "gt_mask": self.masks[index][0],
                'gt_normal': self.normals[index],
                'intrinsic': self.camIntrinsic,
                'extrinsic': self.camExtrinsic,
                # 'depth': depth,
                # "pose_2d": pose_2d,
                # "time": time_encoding(t, self.imgs[index].dtype, self.max_freq),
                "time": index,
                'img_path': self.img_lists[index],
                'index': index,
                # 'tz': depth[depth > 0].mean(),
                ## GT
                # 'gt_vertices': vertices,
                # 'gt_joints': joints_3d,
                # 'height': height
                }
        # if self.opt.has_depth:
        #     data['depth'] = depth
        #     data['tz'] = depth[depth > 0].mean()
        return data


def my_collate_fn(batch):
    camera_params = batch[0]['camera_params']
    intrinsic = batch[0]['intrinsic']
    extrinsic = batch[0]['extrinsic']

    for sample in batch:
        del sample['camera_params']
        del sample['intrinsic']
        del sample['extrinsic']

    try:
        collated_batch = torch.utils.data.dataloader.default_collate(batch)
    except:
        print()

    return {
        'camera_params': camera_params,
        'intrinsic': intrinsic,
        'extrinsic': extrinsic,
        **collated_batch
    }


# def my_collate_fn(batch):
#     return batch[0]


class MMPeopleSnapshotDataModule(pl.LightningDataModule):
    def __init__(self, num_workers, opt, train=True, **kwargs):
        super().__init__()
        if train:
            splits = ["train", "val"]
        else:
            splits = ["test"]
        for split in splits:
            print(f"loading {split}set...")
            dataset = MMDataset(
                opt.dataroot, opt.max_freq, split, opt.get(split), opt.camera, kwargs.get('gender'))
            setattr(self, f"{split}set", dataset)
        self.num_workers = num_workers

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            return DataLoader(self.trainset,
                              shuffle=False,
                              pin_memory=True,
                              batch_size=1,
                              persistent_workers=True,
                              num_workers=self.num_workers,
                              collate_fn=my_collate_fn
                              )
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valset"):
            return DataLoader(self.valset,
                              shuffle=False,
                              pin_memory=True,
                              batch_size=1,
                              persistent_workers=True,
                              num_workers=self.num_workers,
                              collate_fn=my_collate_fn
                              )
        else:
            return super().val_dataloader()

    def test_dataloader(self):
        if hasattr(self, "testset"):
            return DataLoader(self.testset,
                              shuffle=False,
                              pin_memory=True,
                              batch_size=1,
                              persistent_workers=True,
                              num_workers=self.num_workers,
                              collate_fn=my_collate_fn
                              )
        else:
            return super().test_dataloader()
