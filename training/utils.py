import torch
import os
import re
from torchvision.utils import save_image

from torch.amp import autocast

import tempfile
from PIL import Image


###################### Training helpers ######################

@torch.no_grad()
def generate_and_save_images(generator, text_encoder, dataloader, device,
                             save_folder, use_fixed_z=False, fixed_z=None,
                             return_prompts=False):
    """
    Generates images from text and saves them to disk.
    Uses AMP for faster inference. Optionally returns the list of prompts used (just for test).
    """
    generator.eval()
    text_encoder.eval()

    os.makedirs(save_folder, exist_ok=True)
    img_idx = 0
    collected_prompts = []

    for real_imgs, texts in dataloader:
        current_batch_size = real_imgs.size(0)

        if return_prompts:
            collected_prompts.extend(texts)

        # Noise
        if use_fixed_z and fixed_z is not None:
            noise = fixed_z[:current_batch_size].to(device)
        else:
            noise = torch.randn(current_batch_size, generator.z_dim).to(device)

        # Encode text
        with autocast(device_type='cuda'):
            per_token_emb, global_emb = text_encoder(texts)
            fake_imgs = generator(noise, per_token_emb, global_emb)

        # Save images
        for j in range(current_batch_size):
            img = fake_imgs[j]
            img = (img + 1) / 2  # Scale from [-1,1] to [0,1]
            save_path = os.path.join(save_folder, f"sample_{img_idx:05d}.png")
            save_image(img, save_path)
            img_idx += 1

    generator.train()
    text_encoder.train()

    if return_prompts:
        return collected_prompts

###################### Checkpoint helpers ######################

def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(generator, discriminator, text_encoder, optimizerG, optimizerD, scalerG, scalerD, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    optimizerG.load_state_dict(checkpoint["optimizerG"])
    optimizerD.load_state_dict(checkpoint["optimizerD"])
    scalerG.load_state_dict(checkpoint["scalerG"])
    scalerD.load_state_dict(checkpoint["scalerD"])

    start_epoch = checkpoint["epoch"]
    fixed_z = checkpoint["fixed_z"]
    best_fid = checkpoint.get("fid_score", float("inf"))
    global_step = checkpoint.get("global_step")
    print(f"Resumed from checkpoint: {checkpoint_path} at epoch {start_epoch}")
    return start_epoch, fixed_z, best_fid, global_step

def find_latest_checkpoint(checkpoint_dir):
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_epoch_") and f.endswith(".pth")
    ]
    if not checkpoints:
        return None

    # Extract epoch number from filenames like "checkpoint_epoch_25.pth"
    checkpoints = sorted(checkpoints, key=lambda x: int(re.findall(r"\d+", x)[-1]))
    return os.path.join(checkpoint_dir, checkpoints[-1])

###################### Evaluation helpers ######################

def gather_images_from_subfolders(folder):
    all_images = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))
    return all_images

def prepare_flat_real_images_dir(input_folder, output_folder, size=128):
    """
    Flattens and resizes real images from input_folder to output_folder.

    Skips processing if output_folder already contains files.
    """
    if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
        print(f"Using cached resized images in: {output_folder}")
        return

    print(f"Processing real images from {input_folder} to {output_folder} ...")
    os.makedirs(output_folder, exist_ok=True)
    image_paths = gather_images_from_subfolders(input_folder)

    for idx, path in enumerate(image_paths):
        with Image.open(path) as img:
            img = img.convert("RGB") # Ensure 3 channels
            img = img.resize((size, size), Image.LANCZOS)
            save_path = os.path.join(output_folder, f"{idx:05d}.png") # Ensure PNG format
            img.save(save_path, format="PNG")
