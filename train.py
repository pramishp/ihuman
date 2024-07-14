from animatableGaussian.model.nerf_model import NeRFModel
import hydra
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

@hydra.main(config_path="./confs", config_name="mmpeoplesnapshot_fine", version_base="1.1")
def main(opt):
    pl.seed_everything(0)
    optimize_pose = opt.optimize_pose
    datamodule = hydra.utils.instantiate(opt.dataset)
    kwargs = {}
    if optimize_pose:
        kwargs['datamodule'] = datamodule
    model = NeRFModel(opt, **kwargs)

    tensorboard = pl_loggers.TensorBoardLogger(save_dir=opt.log_save_dir, max_queue=10)

    trainer = pl.Trainer(accelerator='gpu',
                         logger=tensorboard,
                         **opt.trainer_args)

    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint('model.ckpt')

    # model = NeRFModel.load_from_checkpoint(checkpoint_path='model.ckpt')
    datamodule = hydra.utils.instantiate(opt.dataset, train=False)
    result = trainer.test(model, datamodule=datamodule)[0]
    print(result)


if __name__ == "__main__":
    main()
