import datetime
import os
import warnings

import hydra
import omegaconf
import pytorch_lightning as pl
from loguru import logger

import scripts.training.lightning_modules as lm


@hydra.main(
    config_path=os.path.join(os.getcwd(), "configs"), config_name="train_experiment"
)
@logger.catch
def main(configs: omegaconf.DictConfig) -> None:
    logger.add("train.log")
    logger.info("๐ Training process started.")

    logger.info("๐ Creating dataset module.")
    dataset_module = lm.DataModuleWithEdges(
        **configs.dataset_module,
        nn_input_image_resolution=configs.train_module.nn_input_image_resolution,
    )

    logger.info("๐ธ Creating training module.")
    train_module = lm.TrainingModuleWithEdges(**configs.train_module)

    logger.info("๐ฒ Initializing callbacks.")
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        **configs.training.early_stop
    )

    model_ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor=configs.training.early_stop.monitor,
        mode=configs.training.early_stop.mode,
        # fmt: off
        filename=configs.training.wandb_name + "-{epoch}-{" + configs.training.early_stop.monitor + ":.4f}",
        # fmt: on
        save_top_k=3,
        dirpath="./models",
        save_last=True,
    )

    logger.info("๐ Initializing W&B logger.")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb_logger = pl.loggers.WandbLogger(
        project=configs.training.wandb_project,
        name=f"{configs.training.wandb_name}-{timestamp}",
    )

    logger.info("๐ Initializing training environment.")
    nn_trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[early_stop_callback, model_ckpt_callback],
        max_epochs=configs.training.max_epochs,
        gpus=1 if configs.training.with_gpu else 0,
    )

    logger.info("๐คน Starting training loop.")
    nn_trainer.fit(train_module, dataset_module)

    logger.info(f"๐ฅ Best model: {model_ckpt_callback.best_model_path}")
    train_module.load_from_checkpoint(model_ckpt_callback.best_model_path)

    logger.info("๐งช Starting testing loop.")
    nn_trainer.test(train_module, dataset_module)

    logger.success("๐ Training process finished.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    main()
