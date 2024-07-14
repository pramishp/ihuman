from typing import List

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra
import numpy as np
import pytorch_lightning as pl
import os
from torch.cuda.amp import custom_fwd
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from animatableGaussian.model.body_model_param import SMPLParamEmbedding
from animatableGaussian.utils import ssim, l1_loss, GMoF, SavePly, save_image

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


class EvaluatorRecon(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""

    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):
        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }


class EvaluatorPose(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""

    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):
        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }


SMPL_TO_BODY25 = [
    24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 2, 28, 29,
    30, 31, 32, 33, 34
]

HIGH_PRIORITY_PARTS = [3, 4, 6, 7, 11, 22, 14, 19, 10, 13]
LOW_PRIORITY_PARTS = [9, 12]


def rgb2shs(rgbs):
    C0 = 1 / (np.sqrt(4 * np.pi))
    return (rgbs - 0.5) / C0


class NeRFModel(pl.LightningModule):
    def __init__(self, opt, datamodule=None):
        super(NeRFModel, self).__init__()
        if not opt.get('optimize_pose', False):
            self.save_hyperparameters()
        self.opt = opt
        self.datamodule = datamodule
        if datamodule is not None:
            self.SMPL_param = SMPLParamEmbedding(**datamodule.trainset.get_SMPL_params())
        self.model = hydra.utils.instantiate(opt.deformer)
        self.training_args = opt.training_args
        self.sh_degree = opt.max_sh_degree
        self.lambda_dssim = opt.lambda_dssim
        self.evaluator = EvaluatorRecon()
        if not os.path.exists("val"):
            os.makedirs("val")
        if not os.path.exists("test"):
            os.makedirs("test")
        self.robustifier = GMoF(rho=5)

        self.cal_test_metrics = opt.cal_test_metrics if 'cal_test_metrics' in opt else False

    def forward(self, camera_params, model_param, time, render_point=False, train=True):

        # if self.opt.use_human_rasterizer:
        #     pass
        # else:
        #     pass
        # from diff_gaussian_rasterization import Gaussian RasterizationSettings, GaussianRasterizer

        is_use_ao = (not train) or self.current_epoch > 3

        verts, opacity, scales, rotations, shs, transforms, verts_posed, normals = self.model(time=time,
                                                                                                   is_use_ao=is_use_ao,
                                                                                                   **model_param)

        means2D = torch.zeros_like(
            verts, dtype=verts.dtype, requires_grad=False, device=verts.device)
        try:
            means2D.retain_grad()
        except:
            pass
        raster_settings = GaussianRasterizationSettings(
            sh_degree=self.sh_degree,
            prefiltered=False,
            debug=False, **camera_params
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        cov3D_precomp = None
        if render_point:
            colors_precomp = torch.rand_like(scales)
            scales /= 10
            opacity *= 100
            shs = None
        else:
            colors_precomp = None

        image, radii, depth, alpha = rasterizer(
            means3D=verts,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        # get normal map
        normal_map = None
        if train:
            normalized_f_normals = F.normalize(normals)
            normalized_f_normals = (normalized_f_normals + 1) / 2

            # match the coordinate system
            normalized_f_normals[:, 1] = 1 - normalized_f_normals[:, 1]
            normalized_f_normals[:, 2] = 1 - normalized_f_normals[:, 2]

            # rgb to shs

            normal_shs = rgb2shs(normalized_f_normals).reshape(-1, 1, 3)

            normal_raster_settings = GaussianRasterizationSettings(
                sh_degree=0,
                prefiltered=False,
                debug=False, **camera_params
            )
            normal_rasterizer = GaussianRasterizer(raster_settings=normal_raster_settings)
            normal_map, _, _, _ = normal_rasterizer(means3D=verts,
                                                    means2D=means2D,
                                                    shs=normal_shs,
                                                    colors_precomp=colors_precomp,
                                                    opacities=opacity,
                                                    scales=scales,
                                                    rotations=rotations,
                                                    cov3D_precomp=cov3D_precomp)

            # import cv2
            # cv2.imwrite('normal.png', normal.detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1] * 255.0)

        return image, verts_posed, scales, rotations, depth, alpha, normal_map

    def batch2single_model_param(self, model_params, index):
        body_model_param = {}
        for k in model_params:
            if k == 'betas':
                body_model_param[k] = model_params[k][0].unsqueeze(0)
            else:
                body_model_param[k] = model_params[k][index].unsqueeze(0)
        return body_model_param

    def get_model_params(self, idx: torch.Tensor):
        '''
        Args:
            idx: Tensor of ints

        Returns: params
        '''
        model_params = self.SMPL_param(idx)
        params = {}
        for k, v in model_params.items():
            if k == 'body_pose':
                v = v.reshape(-1, 23, 3)
            params[k] = v
        return params

    def get_current_stage(self):
        hparams = self.opt.training_args
        stages = hparams.stages
        current_stage = None
        for stage_key, stage in stages.items():
            if stage['start_epoch'] <= self.current_epoch:
                current_stage = stage
        if current_stage is None:
            raise Exception(f'No stage is found for epoch {self.current_epoch}. Please check configs.')
        return current_stage

    def training_step(self, batch, batch_idx):

        camera_params = batch["camera_params"]
        gt_images = batch["gt"]

        camera = {
            'intrinsic': batch['intrinsic'],
            'extrinsic': batch['extrinsic'],
            'height': camera_params['image_height'],
            'width': camera_params['image_width']
        }

        # stage based loss weights
        stage = self.get_current_stage()
        smplify_loss_weight = stage["smplify_loss_weight"]
        pose_loss_weight = stage["pose_loss_weight"]
        gaussian_loss_weight = stage["gaussian_loss_weight"]
        mesh_loss_weight = stage['mesh_loss_weight']
        pose_prior_loss_weight = stage['pose_prior_loss_weight']
        scale_loss_weight = stage['scale_loss_weight']
        rotation_loss_weight = stage['rotation_loss_weight']
        normal_loss_weight = stage['normal_loss_weight']
        mask_loss_weight = stage['mask_loss_weight']

        # # ================= optimize pose temporally ==============
        # if self.current_epoch == 0:
        #     optimize_params(camera, keypoints=pose_2d,
        #                     model_params_embedding=self.SMPL_param,
        #                     get_model_param=self.get_model_params,
        #                     model=self.model,
        #                     index=batch['index'],
        #                     max_iter=50)

        # adjust learning rate according to epoch
        if batch_idx == 0:
            self.adjust_learning_rate(self.current_epoch)

        # ========== Model Params ==============
        if self.datamodule is not None:
            model_params = self.get_model_params(batch["index"])
        else:
            model_params = batch['model_param']

        # =============== LOSS ============================

        mse = nn.MSELoss()

        # ================= Smplify Loss =====================
        smplify_loss = 0
        if smplify_loss_weight > 0:
            from animatableGaussian.smplify.optimize_pose import smplify, optimize_params
            # gt 2d pose
            pose_2d = batch["pose_2d"]
            # TODO: fix model_params for smplify
            optimize_model_params = smplify(camera, pose_2d, model_params,
                                            self.model, max_iter=100)

            ### ============= Loss computation ================== ####

            ## smplify loss
            body_pose_loss = mse(model_params['body_pose'], optimize_model_params['body_pose'])
            transl_loss = mse(model_params['transl'], optimize_model_params['transl'])
            global_orient_loss = mse(model_params['global_orient'], optimize_model_params['global_orient'])

            smplify_loss = (global_orient_loss + body_pose_loss + transl_loss).sum()

        # =========== pose loss ==========
        pose_loss = 0
        if pose_loss_weight > 0:
            # gt 2d pose
            pose_2d = batch["pose_2d"]
            # gaussian to joints
            pred_pose2d = self.model.get_2d_pose(camera, **model_params)
            error = (pose_2d[..., :2] - pred_pose2d).square().sum(-1).sqrt()
            m1 = (pose_2d[..., 2] > 0.2)
            error = error * m1.float()

            ### priority over parts
            error_weights = torch.ones_like(error)
            # error_weights[HIGH_PRIORITY_PARTS] *= 50 * 10
            # error_weights[LOW_PRIORITY_PARTS] *= 0.5

            weighted_error = error * error_weights

            pose_loss = weighted_error.mean()
        # ===================

        ## guassian spolatting loss and depth loss

        rasterized_rgbs, b_vertices, b_scales, b_rotations, b_depths, b_alphas, b_normals = [], [], [], [], [], [], []
        gaussian_loss = 0
        if gaussian_loss_weight > 0:
            # iterate through each params and pass through gaussian rasterization
            for i in range(len(batch['gt'])):
                body_model_param = self.batch2single_model_param(model_params, i)
                time = batch['time'][i]
                rgb, vt, scales, rotations, depth, alpha, normal = self(camera_params, body_model_param, time)
                rasterized_rgbs.append(rgb)
                b_vertices.append(vt)
                b_scales.append(scales)
                b_rotations.append(rotations)
                b_depths.append(depth)
                b_alphas.append(alpha)
                b_normals.append(normal)

            # convert list to torch tensors
            rasterized_rgbs = torch.stack(rasterized_rgbs)
            b_vertices = torch.stack(b_vertices)
            b_scales = torch.stack(b_scales)
            b_rotations = torch.stack(b_rotations)
            b_depths = torch.stack(b_depths)
            b_alphas = torch.stack(b_alphas)
            b_normals = torch.stack(b_normals)

            Ll1 = l1_loss(
                rasterized_rgbs, gt_images)
            gaussian_loss = (1.0 - self.lambda_dssim) * Ll1 + \
                            self.lambda_dssim * (1.0 - ssim(rasterized_rgbs, gt_images))

        mesh_loss = 0
        if mesh_loss_weight > 0:
            # normal loss
            from pytorch3d.structures import Meshes
            from pytorch3d.loss import mesh_normal_consistency, mesh_edge_loss, mesh_laplacian_smoothing

            meshes = Meshes(verts=b_vertices, faces=self.model.faces.repeat((b_vertices.shape[0], 1, 1)))
            loss_normal = mesh_normal_consistency(meshes)
            loss_edge = mesh_edge_loss(meshes)
            loss_laplacian = mesh_laplacian_smoothing(meshes, method="uniform")

            # mesh_loss = loss_normal + loss_edge + loss_laplacian
            mesh_loss = loss_normal + loss_edge

        # ========= Pose Prior Loss =============
        pose_prior_loss = 0
        if pose_prior_loss_weight > 0:
            # pose prior loss
            t_minus_1_idx = torch.where(batch['index'] > 0, batch['index'] - 1, 0)
            t_idx = batch['index']

            t_minus_1_model_params = self.get_model_params(t_minus_1_idx)
            t_model_params = self.get_model_params(t_idx)

            t_body_pose_loss = l1_loss(t_model_params['body_pose'], t_minus_1_model_params['body_pose'].detach())
            t_transl_loss = l1_loss(t_model_params['transl'], t_minus_1_model_params['transl'].detach())
            # t_global_orient_loss = l1_loss(t_model_params['global_orient'],
            #                                t_minus_1_model_params['global_orient'].detach())

            pose_prior_loss = t_body_pose_loss + t_transl_loss

        # ============= Scale Regularization ===============

        scale_loss = 0
        if scale_loss_weight > 0:
            min_scal_comp, _ = torch.min(b_scales, dim=2)
            scale_loss = torch.mean(min_scal_comp.abs())

        rotation_loss = 0
        if rotation_loss_weight > 0:
            from pytorch3d.transforms import quaternion_to_matrix
            R = quaternion_to_matrix(torch.nn.functional.normalize(self.model.rotations))
            # R = R.squeeze(0)  # TODO: support batch
            min_indices = torch.argmin(b_scales, dim=2)
            n_c = F.one_hot(min_indices, num_classes=3).float()
            n_c = n_c.squeeze(0)  # TODO: support batch
            n_w = torch.bmm(R, n_c.unsqueeze(2)).squeeze(2)

            dot_product = torch.sum(n_w * self.model.vertex_normals, dim=1, keepdim=True)
            rotation_loss = (1 - dot_product).abs().mean()

        normal_loss = 0
        if normal_loss_weight > 0:
            def save_img(img, mask, fname='normal.png'):
                import cv2
                img = torch.where(mask, img, 0)
                cv2.imwrite(fname, img[0].detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1] * 255.0)

            # mask = batch['gt_mask']
            gt_normals = batch['gt_normal']

            # normal_loss = nn.MSELoss()(b_normals, gt_normals)

            # dot_product = torch.sum(masked_pred_normal.reshape(-1, 3) * gt_normal.reshape(-1, 3), dim=1, keepdim=True)
            # mask_dot_product = torch.arccos(dot_product) < (torch.pi * 60 / 180)
            # normal_loss = (1 - dot_product[mask_dot_product]).abs().mean()
            #
            # pixel_dist = (masked_pred_normal - gt_normal).abs()
            # normal_mask1 = pixel_dist[:, 0, :, :] < 0.3
            # normal_mask2 = pixel_dist[:, 1, :, :] < 0.3
            # # normal_mask3 = pixel_dist[:, 2, :, :] < 0.3
            # normal_mask = normal_mask1 & normal_mask2

            # save_img(gt_normal, normal_mask, 'gt.png')
            # save_img(masked_pred_normal, normal_mask, 'pred.png')
            #
            # normal_mask = torch.stack([normal_mask, normal_mask]).permute(1, 0, 2, 3)
            # normal_loss = F.l1_loss(masked_pred_normal[:, [0, 1], :, :][normal_mask],
            #                         gt_normal[:, [0, 1], :][normal_mask])

            # normal_loss = F.l1_loss(masked_pred_normal[:, [0,1], :, :], gt_normal[:, [0,1], :, :])
            # distances = torch.sum((masked_pred_normal.reshape(-1, 3) - gt_normal.reshape(-1, 3)).square(), dim=1).sqrt()
            # normal_mask = distances < 100
            # normal_loss = distances[normal_mask].mean()
            # normal_loss = self.robustifier(distances)
            # normal_loss = F.l1_loss(b_normals, gt_normals, reduction='mean')
            # normal_loss = 1 - F.cosine_similarity(b_normals, gt_normals).mean()
            # print()

            # normals_pred, normals_gt = b_normals[:, [0, 1], :, :], gt_normals[:, [0, 1], :, :]
            normals_pred, normals_gt = b_normals[:, :, :, :], gt_normals[:, :, :, :]
            Ll1 = l1_loss(
                normals_pred, normals_gt)
            normal_loss = (1.0 - self.lambda_dssim) * Ll1 + \
                          self.lambda_dssim * (1.0 - ssim(normals_pred, normals_gt))
            # normal_loss = Ll1

        mask_loss = 0
        if mask_loss_weight > 0:
            mask_loss = F.mse_loss(b_alphas, batch['gt_mask'])

        # gaussian_loss_weight = 0
        loss = gaussian_loss_weight * gaussian_loss + \
               smplify_loss_weight * smplify_loss + \
               pose_loss_weight * pose_loss + \
               mesh_loss_weight * mesh_loss + \
               pose_prior_loss_weight * pose_prior_loss + \
               scale_loss_weight * scale_loss + \
               rotation_loss_weight * rotation_loss + \
               normal_loss_weight * normal_loss + \
               mask_loss_weight * mask_loss

        self.log('train_loss', loss, prog_bar=True)
        self.log('gaussian_loss', gaussian_loss_weight * gaussian_loss, prog_bar=True)
        # self.log('pose_loss', pose_loss, prog_bar=True)
        # self.log('smplify_loss', smplify_loss, prog_bar=True)
        self.log('mesh_loss', mesh_loss, prog_bar=True)
        # self.log('pose_prior_loss', pose_prior_loss, prog_bar=True)
        # self.log('scale_loss', scale_loss * scale_loss_weight, prog_bar=True)
        # self.log('rotation_loss', rotation_loss * rotation_loss_weight, prog_bar=True)
        self.log('normal_loss', normal_loss_weight * normal_loss, prog_bar=True)

        self.log('model_scale', self.model.model_scale)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # if batch['index'] > 15:
        # if batch['index'] % 15 != 0:
        #     return
        batch_size = batch['gt'].shape[0]
        gt_images = batch["gt"]
        gt_normals = batch['gt_normal']

        # camera
        camera_intrinsic = batch['intrinsic']
        camera_extrinsic = batch['extrinsic']

        camera_params = batch["camera_params"]
        camera = {
            'intrinsic': camera_intrinsic,
            'extrinsic': camera_extrinsic,
            'height': camera_params['image_height'],
            'width': camera_params['image_width']
        }

        if self.datamodule is not None:
            model_params = self.get_model_params(batch['index'])
        else:
            model_params = batch['model_param']

        # optimize_model_params = smplify(camera, pose_2d, model_param, self.model)
        # model_param = optimize_model_params
        ### ==== Create output folder for epoch ==========
        if not os.path.exists(f"val/{self.current_epoch}"):
            os.mkdir(f"val/{self.current_epoch}")
        # =========================

        # ======= optimize poses ==============
        # optimize_model_params = smplify(camera, pose_2ds, model_params, self.model, max_iter=200)
        # model_params = optimize_model_params

        ## ========= gaussian rasterization ==============

        rasterized_rgbs, b_vertices, b_normals = [], [], []

        # iterate through each params and pass through gaussian rasterization
        for i in range(len(batch['gt'])):
            body_model_param = self.batch2single_model_param(model_params, i)
            time = batch['time'][i]
            rgb, vt, scales, rotation, depth, alpha, normal = self(camera_params, body_model_param, time)

            rasterized_rgbs.append(rgb)
            b_vertices.append(vt)
            b_normals.append(normal)

        # convert list to torch tensors
        rasterized_rgbs = torch.stack(rasterized_rgbs)
        b_vertices = torch.stack(b_vertices)
        b_normals = torch.stack(b_normals)

        from animatableGaussian.vis_utils import create_side_by_side_images
        gt_rgb_vs_rasterized_rgb = create_side_by_side_images(gt_images=gt_images, pred_images=rasterized_rgbs)

        save_image(gt_rgb_vs_rasterized_rgb * 255,
                   path=f"val/{self.current_epoch}/rasterized_{batch_idx}.png")

        gt_normal_vs_rasterized_normal = create_side_by_side_images(gt_images=gt_normals, pred_images=b_normals)
        save_image(gt_normal_vs_rasterized_normal * 255,
                   path=f"val/{self.current_epoch}/normal_{batch_idx}.png")

        ## ===================================================

        ## ================= save as point cloud ====================
        verts, opacity, scales, rotations, shs, transforms, _, _ = self.model(time=batch['time'], is_use_ao=False,
                                                                                   **model_params)
        #
        # # save a point cloud
        from animatableGaussian.utils import save_ply
        for i, vt in enumerate(b_vertices):
            save_ply(vt, f"val/{self.current_epoch}/pc_{batch_idx}_{i}.ply")

        # # save a mesh
        # from animatableGaussian.utils import save_mesh
        # for i, vt in enumerate(b_vertices):
        #     save_mesh(vt, self.model.faces[0], f"val/{self.current_epoch}/mesh_{batch_idx}_{i}.obj")

        # save pc with full gaussian information
        # TODO: save gaussian
        # ply = SavePly(verts, opacity, scales, rotations, shs)
        # ply.save_ply(f"val/{self.current_epoch}_{batch_idx}_full.ply")

        # ===================================================

        ######## pose visualization ###############3
        #
        # pred_pose3d = self.model.get_joints_from_pose(body_pose=model_params['body_pose'],
        #                                               global_orient=model_params['global_orient'],
        #                                               transl=model_params['transl'])
        #
        # ### visualize 3d joints ###
        # from animatableGaussian.vis_utils import visualize_pose_3d
        # # visualize_pose_3d(pred_pose3d[0].detach().cpu().numpy()[SMPL_TO_BODY25])
        #
        # pred_poses2d = self.model.get_2d_pose(camera, body_pose=model_params['body_pose'],
        #                                       global_orient=model_params['global_orient'],
        #                                       transl=model_params['transl'])
        #
        # # ### optimized pose smplify ####
        # # optimized_poses_2d = self.model.get_2d_pose(camera, body_pose=optimize_model_params['body_pose'],
        # #                                             global_orient=optimize_model_params['global_orient'],
        # #                                             transl=optimize_model_params['transl'])
        #
        # if 'pose_2d' in batch:
        #     pose_2d_gts = batch["pose_2d"]
        #
        #     from animatableGaussian.vis_utils import create_pose_comparison_image
        #     pose_image_grid = create_pose_comparison_image(gt_images, rasterized_rgbs, gt_poses=pose_2d_gts,
        #                                                    model_poses=pred_poses2d,
        #                                                    # optimized_poses=vibe_poses_2d
        #                                                    )
        #
        #     save_image(pose_image_grid, f"val/{self.current_epoch}/pose_{batch_idx}.png")

        ## ========================================== ###

        # ==================== render model and image ===============
        from animatableGaussian.vis_utils import render_model_to_image

        model_image_overlaps = []
        for i, img_path in enumerate(batch['img_path']):
            model_img_overlap = render_model_to_image(b_vertices[i].detach().cpu().unsqueeze(0).numpy(),
                                                      camera,
                                                      [img_path],
                                                      save_path=None)
            model_image_overlaps.append(torch.from_numpy(model_img_overlap))

        model_image_overlaps = torch.stack(model_image_overlaps).permute(0, 3, 1, 2)

        from animatableGaussian.vis_utils import make_grid
        grid_model_image_overlap = make_grid(model_image_overlaps)
        save_image(grid_model_image_overlap, f'val/{self.current_epoch}/model_over_rgb_{batch_idx}.png')

        # =====
        # model front side
        from animatableGaussian.vis_utils import render_model_front_n_side
        front_side_model_views = []
        for i, img_path in enumerate(batch['img_path']):
            front_side_model_view = render_model_front_n_side(b_vertices[i].detach().cpu().unsqueeze(0).numpy(),
                                                              camera)
            front_side_model_views.append(torch.from_numpy(front_side_model_view))

        front_side_model_views = torch.stack(front_side_model_views).permute(0, 3, 1, 2)

        from animatableGaussian.vis_utils import make_grid
        grid_front_side = make_grid(front_side_model_views)
        save_image(grid_front_side, f'val/{self.current_epoch}/front_side_view_model_{batch_idx}.png')

        # log to tensorflow
        tensorboard = self.logger.experiment
        if tensorboard:

            tensorboard.add_image(f'rgb_reconstructed_{batch_idx}',
                                  gt_rgb_vs_rasterized_rgb,
                                  self.current_epoch, dataformats='HWC')

            # if 'pose_2d' in batch:
            #     tensorboard.add_image(f'pose_{batch_idx}',
            #                           pose_image_grid / 255.0,
            #                           self.current_epoch, dataformats='HWC')

            tensorboard.add_image(f'model_img_overlap_{batch_idx}',
                                  grid_model_image_overlap / 255.0,
                                  self.current_epoch, dataformats='HWC')

            tensorboard.add_image(f'model_front_side_{batch_idx}',
                                  grid_front_side / 255.0,
                                  self.current_epoch, dataformats='HWC')

            tensorboard.add_image(f'normal_{batch_idx}', gt_normal_vs_rasterized_normal, self.current_epoch,
                                  dataformats="HWC")

            # mesh visualization
            camera_config = {'cls': 'PerspectiveCamera'}
            verts = b_vertices.cpu().clone()
            verts[:, :, 1] *= -1
            verts -= verts.mean(1).unsqueeze(1)

            faces = self.model.faces[None, ...]

            pc = verts.clone()

            # pred_gt_verts = verts.unsqueeze(0)
            # append GT verts
            if self.cal_test_metrics and 'gt_vertices' in batch.keys():
                gt_verts = batch['gt_vertices'].detach().cpu()
                gt_verts -= gt_verts.mean(1).unsqueeze(1)
                gt_verts[:, :, 1] *= -1
                gt_verts[:, :, 0] += 1
                pc = torch.hstack([verts, gt_verts])

            tensorboard.add_mesh(f'reconstructed_pc_{batch_idx}',
                                 vertices=pc,
                                 # faces=faces,
                                 config_dict={"camera": camera_config},
                                 global_step=self.current_epoch)

    @torch.no_grad()
    def test_step(self, batch, batch_idx, *args, **kwargs):
        camera_params = batch["camera_params"]
        model_param = batch["model_param"]
        rgb, vt, scales, rotation, depth, alpha, normal = self(camera_params, model_param, batch["time"], train=False)
        rgb_gt = batch["gt"][0]
        losses = {
            # add some extra loss here
            **self.evaluator(rgb[None], rgb_gt[None]),
            "rgb_loss": (rgb - rgb_gt).square().mean(),
        }
        image = rgb
        img = (255. * image.permute(1, 2, 0)
               ).data.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"test/{batch_idx}.png")
        image = rgb_gt
        img = (255. * image.permute(1, 2, 0)
               ).data.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"test/{batch_idx}_gt.png")

        for k, v in losses.items():
            self.log(f"test/{k}", v, on_epoch=True, batch_size=1)

        # save predicted mesh
        if batch_idx == 0:
            # save mesh
            f_displaced, opacity, scales, quaternions, \
                shs, T, verts_posed, faces_normals = self.model(**model_param, time=0)

            pred_faces = self.model.faces[0]

            import open3d as o3d

            def too3dmesh(vertx, triangles, colors=None):

                # Convert vertex positions to Open3D format
                verts_posed_o3d = o3d.utility.Vector3dVector(vertx.cpu().numpy())
                # Convert faces to Open3D format
                faces_o3d = o3d.utility.Vector3iVector(triangles.cpu().numpy())
                # Create an Open3D TriangleMesh object
                mesh_o3d = o3d.geometry.TriangleMesh()
                # Set vertex positions and vertex colors
                mesh_o3d.vertices = verts_posed_o3d
                mesh_o3d.triangles = faces_o3d
                if colors is not None:
                    vertex_colors_o3d = o3d.utility.Vector3dVector(colors.cpu().numpy())
                    mesh_o3d.vertex_colors = vertex_colors_o3d
                # Save the mesh
                return mesh_o3d

            mesh_o3d = too3dmesh(verts_posed, pred_faces)
            o3d.io.write_triangle_mesh(f"test/mesh.obj", mesh_o3d)
            #
            # # # save colored mesh
            vertex_colors = self.model.get_vertex_colors()
            mesh_o3d = too3dmesh(verts_posed, pred_faces, vertex_colors)

            o3d.io.write_triangle_mesh(f"test/mesh_colored.obj", mesh_o3d)

        return losses

    def adjust_learning_rate(self, current_epoch):
        hparams = self.opt.training_args
        stages = hparams.stages
        stage_params = None
        for stage_key, stage in stages.items():
            if stage['start_epoch'] == current_epoch:
                stage_params = stage
        if stage_params is None:
            return
        # Get optimizers (handle both single and multiple optimizers)
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        # Update learning rates of all optimizers
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_name = param_group.get('name', 'default')
                new_lr = stage_params.get(param_name + '_lr', None)
                if new_lr is not None:
                    param_group['lr'] = new_lr


    def configure_optimizers(self):
        l = []

        # add smpl params
        body_pose_params = []
        global_orient_params = []
        transl_params = []
        for (name, param) in self.named_parameters():
            if name.startswith("SMPL_param.body_pose"):
                body_pose_params.append(param)
            elif name.startswith("SMPL_param.global_orient"):
                global_orient_params.append(param)
            elif name.startswith("SMPL_param.transl"):
                transl_params.append(param)
            else:
                pass

        if self.opt.optimize_pose:
            # get from opt
            body_pose_lr = self.opt.training_args.body_pose_lr
            global_orient_lr = self.opt.training_args.global_orient_lr
            transl_lr = self.opt.training_args.transl_lr
            # lr 9e-3
            l.append(
                {"params": body_pose_params, "lr": body_pose_lr, "name": "body_pose"})  # TODO: get learning rate from config
            l.append({"params": global_orient_params, "lr": global_orient_lr,
                      "name": "global_orient"})  # TODO: get learning rate from config
            l.append({"params": transl_params, "lr": transl_lr, "name": "transl"})  # TODO: get learning rate from config
            # l.append({"params": body_pose_params, "lr": 0, "name": "body_pose"})  # TODO: get learning rate from config
            # l.append({"params": global_orient_params, "lr": 0,
            #           "name": "global_orient"})  # TODO: get learning rate from config
            # l.append({"params": transl_params, "lr": 0, "name": "transl"})  # TODO: get learning rate from config

        return self.model.configure_optimizers(self.training_args, extra_params=l)
