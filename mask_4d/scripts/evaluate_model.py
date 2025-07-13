import os
from os.path import join
from pathlib import Path

import click
import torch
import yaml
from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything

from mask_4d.datasets.kitti_dataset import SemanticDatasetModule
from mask_4d.models.mask_model import Mask4D


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


@click.command()
@click.option(
    "--checkpoint-filepath",
    "-f",
    type=str,
    required=True,
    default=str(Path(__file__).parents[2] / "mask4d.ckpt"),
)
@click.option(
    "--save_testset",
    is_flag=True,
)
def main(checkpoint_filepath: str, save_testset: bool):
    seed_everything(42, workers=True)
    model_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml"))))
    backbone_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml"))))
    decoder_cfg = edict(yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml"))))
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})

    cfg.EVALUATE = True
    if save_testset:
        results_dir = create_dirs()
        print(f"Saving test set predictions in directory {results_dir}")
        cfg.RESULTS_DIR = results_dir

    data = SemanticDatasetModule(cfg)
    model = Mask4D(cfg)
    checkpoint = torch.load(checkpoint_filepath, map_location="cpu")
    if torch.__version__ > "2.0":
        new_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            if v.ndim == 5:
                if ".deconv." in k:
                    v = v.permute(3, 0, 1, 2, 4).contiguous()
                elif ".conv." in k:
                    # swap Dim-0 (out) with Dim-3 (D) from old-layout -> new-layout
                    # [D, H, W, out, in] -> [out, D, H, W, in]
                    v = v.permute(3, 0, 1, 2, 4).contiguous()
            new_state_dict[k] = v
        checkpoint["state_dict"] = new_state_dict
    model.load_state_dict(checkpoint["state_dict"])

    trainer = Trainer(
        devices=cfg.TRAIN.N_GPUS,
        limit_val_batches=1,
    )

    if save_testset:
        trainer.test(model, data)
    else:
        trainer.validate(model, data)
    model.evaluator.print_results()
    print("#############################################################")
    model.evaluator4d.print_results()


def create_dirs():
    results_dir = join(getDir(__file__), "..", "output", "test", "sequences")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    for i in range(11, 22):
        sub_dir = os.path.join(results_dir, str(i).zfill(2), "predictions")
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)
    return results_dir


if __name__ == "__main__":
    main()
