from animatableGaussian.model.nerf_model import NeRFModel
import hydra
import pytorch_lightning as pl


@hydra.main(config_path="./confs", config_name="mmpeoplesnapshot_fine", version_base="1.1")
def main(opt):
    pl.seed_everything(0)

    datamodule = hydra.utils.instantiate(opt.dataset, train=False)
    model = NeRFModel.load_from_checkpoint(checkpoint_path='model.ckpt')
    trainer = pl.Trainer(accelerator='gpu',
                         **opt.trainer_args)
    result = trainer.test(model, datamodule=datamodule)[0]
    print(result)


if __name__ == "__main__":
    main()
