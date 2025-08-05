import os
import time
import argparse

import pandas as pd
import yaml
import json

import torch
from torch.utils.tensorboard import SummaryWriter

from data.dataset import create_dataloaders
from data.data_utils import set_all_seeds

from models.text_encoder import TextEncoder
from models.generator import Generator
from models.discriminator import Discriminator

from training.trainer import train_loop
from training.evaluate import evaluate_test_sets
from training.utils import prepare_flat_real_images_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate your text-to-image model")

    ### General ###
    parser.add_argument("--experiment_name", type=str, default="experiment_1", help="Name of the experiment folder")

    parser.add_argument("--train", type=bool, default=False, help="Run training mode")
    parser.add_argument("--resume", type=bool, default=False, help="Resume training from last checkpoint")
    parser.add_argument("--test", type=bool, default=False, help="Run test mode")
    parser.add_argument("--inference", type=bool, default=False)

    ##
    parser.add_argument("--cfg", type=str, default=None, help="Optional config file (yaml or json)")
    parser.add_argument("--image_size", type=int, default=128, help="Image Size")

    ### Dataset ###
    parser.add_argument("--csv_path", type=str, default="./dataset/pokedex.csv", help="Path to CSV file with annotations")
    parser.add_argument("--images_dir", type=str, default="./dataset/images", help="Root directory containing train/val/test folders")

    ###
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    ### Text encoder ###
    parser.add_argument("--encoder_type", type=str, default="bert", help="Type of text encoder")

    ### Generator ###
    parser.add_argument("--up_type", type=str, default="nearest")

    ### Discriminator ###
    parser.add_argument("--use_spectral_norm", type=bool, default=True)
    parser.add_argument("--down_type", type=str, default="stride_conv")

    ### Optimizer ###
    parser.add_argument("--lr_G", type=float, default=1e-4, help="Learning rate for generator")
    parser.add_argument("--lr_D", type=float, default=4e-4, help="Learning rate for discriminator")
    parser.add_argument("--lr_text", type=float, default=1e-5, help="Learning rate for text encoder")


    args = parser.parse_args()
    return args

def load_config_into_args(args):
    if args.cfg is None:
        return args

    if args.cfg.endswith(".yaml") or args.cfg.endswith(".yml"):
        with open(args.cfg, "r") as f:
            cfg_dict = yaml.safe_load(f)
    elif args.cfg.endswith(".json"):
        with open(args.cfg, "r") as f:
            cfg_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {args.cfg}")

    for key, value in cfg_dict.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"Warning: Unknown config key '{key}' in {args.cfg}")
    return args

def main(args):
    # Load config overrides if provided
    args = load_config_into_args(args)

    ### Set seed
    set_all_seeds()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    ### Print and save args
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # Optionally save to yaml if needed
    # save_args_to_yaml(args, os.path.join(args.save_dir, "args.yaml"))

    print("\nLoading dataset ...")
    # Load CSV just keeping necessary columns
    df = pd.read_csv(args.csv_path)

    print("\nPreparing data loaders ...")
    ### Prepare dataloaders
    train_loader, val_loader, intra_test_loader, novel_test_loader = create_dataloaders(
        df=df,
        images_dir=args.images_dir,
        image_size = args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
        )

    print(f"\nTrain Dataset Size: {len(train_loader.dataset)}")
    print(f"Validation Dataset Size: {len(val_loader.dataset)}")
    print(f"Intra Test Dataset Size: {len(intra_test_loader.dataset)}")
    print(f"Novel Test Dataset Size: {len(novel_test_loader.dataset)}")

    print("\nPreparing models and optimizers ...")
    ### Prepare models and optimizers
    generator = Generator(z_dim=128, text_dim=256, image_size=args.image_size, up_type=args.up_type).to(device)
    discriminator = Discriminator(use_spectral_norm=args.use_spectral_norm, down_type=args.down_type, image_size=args.image_size).to(device)
    text_encoder = TextEncoder(encoder_type=args.encoder_type).to(device)

    optimizerG = torch.optim.Adam([
        {"params": generator.parameters(), "lr": args.lr_G},
        {"params": filter(lambda p: p.requires_grad, text_encoder.parameters()), "lr": args.lr_text},
    ], betas=(0.0, 0.9))

    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.lr_D, betas=(0.0, 0.9))

    fixed_z = torch.randn(args.batch_size, 128).to(device)

    ### Train ###
    if args.train:
        print("\nStart training ...")

        # -----------------------------
        # Prepare all necessary folders
        # -----------------------------
        args.checkpoint_dir = os.path.join("runs", args.experiment_name, "checkpoints")
        args.writer_dir = os.path.join("runs", args.experiment_name, "logs")
        val_dir = os.path.join("runs", args.experiment_name, "validation")

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.writer_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        val_real_dir = os.path.join(val_dir, "valid_real")
        prepare_flat_real_images_dir(os.path.join(args.images_dir, "validation"), val_real_dir, size=args.image_size)
        # -----------------------------

        # Tensorboard Logger
        writer = SummaryWriter(log_dir=args.writer_dir)

        start_time = time.time()  # <-- Start timer

        # Start training loop
        train_loop(
            num_epochs=args.num_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            generator=generator,
            discriminator=discriminator,
            text_encoder=text_encoder,
            optimizerG=optimizerG,
            optimizerD=optimizerD,
            fixed_z=fixed_z,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            writer=writer,
            real_val_folder=val_real_dir,
            gen_val_folder=val_dir,
            ckpt_interval=5,
            fid_interval=10,
            resume_checkpoint=args.resume
        )

        end_time = time.time()  # <-- End timer
        elapsed = end_time - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"\nTraining completed in {hours}h {minutes}m {seconds}s.")


    ### Test ###
    if args.test:
        print("\nStart tests ...")

        # -----------------------------
        # Prepare all necessary folders
        # -----------------------------
        args.checkpoint_dir = os.path.join("runs", args.experiment_name, "checkpoints")

        base_test_dir = os.path.join("runs", args.experiment_name, "test")

        intra_dir = os.path.join(base_test_dir, "test_intra")
        novel_dir = os.path.join(base_test_dir, "novel_test")

        intra_real = os.path.join(intra_dir, "test_intra_real")
        intra_gen = os.path.join(intra_dir, "test_intra_gen")

        novel_real = os.path.join(novel_dir, "novel_test_real")
        novel_gen = os.path.join(novel_dir, "novel_test_gen")

        prepare_flat_real_images_dir(os.path.join(args.images_dir, "test_intra"), intra_real, size=args.image_size)
        prepare_flat_real_images_dir(os.path.join(args.images_dir, "test_novel"), novel_real, size=args.image_size)

        results_file = os.path.join(base_test_dir, "test_metrics.txt") # file where evaluation results are saved
        # -----------------------------

        test_sets = {
            "intra-test": (intra_test_loader, intra_real, intra_gen),
            "novel-test": (novel_test_loader, novel_real, novel_gen)
        }

        evaluate_test_sets(
            generator=generator,
            text_encoder=text_encoder,
            test_sets=test_sets,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            use_fixed_z=True,
            results_file=results_file
        )


if __name__ == '__main__':
    args = parse_args()
    main(args)