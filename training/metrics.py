import os
from PIL import Image

import torch
from torchvision import transforms
from torch_fidelity import calculate_metrics
from torchmetrics.multimodal import CLIPScore


def compute_fid(real_folder, gen_folder):
    """
    Compute FID score.
    """
    metrics = calculate_metrics(
        input1=real_folder,
        input2=gen_folder,
        fid=True
    )

    return metrics['frechet_inception_distance']

def compute_inception_score(gen_folder):
    """
    Compute IS.
    """
    metrics = calculate_metrics(
        input1=gen_folder,
        isc=True
    )

    return metrics['inception_score_mean'], metrics["inception_score_std"]


def compute_kid_score(real_folder, gen_folder):
    """
    Compute KID.
    """

    # Count valid images
    num_real = len([f for f in os.listdir(real_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    num_fake = len([f for f in os.listdir(gen_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    kid_subset_size = min(num_real, num_fake)

    # Compute KID
    metrics = calculate_metrics(
        input1=real_folder,
        input2=gen_folder,
        kid=True,
        kid_subset_size=kid_subset_size,
        kid_subsets=100
    )

    return metrics['kernel_inception_distance_mean'], metrics['kernel_inception_distance_std']


def compute_clip_score(gen_folder, prompts, device="cuda", model_name="openai/clip-vit-base-patch32"):
    """
    Compute CLIP Score.
    """
    clip_metric = CLIPScore(model_name_or_path=model_name).to(device)
    clip_metric.eval()

    # Define a transform that loads the image as a PyTorch Tensor
    # with values in the [0, 255] range and uint8 dtype.
    # This is what the default CLIPProcessor expects.
    to_tensor_0_255 = transforms.PILToTensor()

    image_files = sorted(os.listdir(gen_folder))
    total_score = 0.0
    count = 0

    if not isinstance(prompts, (list, tuple)):
        prompts = [prompts] * len(image_files)
    
    if len(image_files) != len(prompts):
        raise ValueError("The number of image files must match the number of prompts.")

    for img_file, prompt in zip(image_files, prompts):
        img_path = os.path.join(gen_folder, img_file)

        # Convert PIL Image to PyTorch Tensor with values 0-255        
        pil_image = Image.open(img_path).convert("RGB")
        image_tensor = to_tensor_0_255(pil_image).to(device)
        
        # Add a batch dimension (N, C, H, W)
        image_tensor = image_tensor.unsqueeze(0) 

        with torch.no_grad():
            # Pass the PyTorch tensor image and the string prompt
            score = clip_metric(image_tensor, prompt)
        
        total_score += score.item()
        count += 1

    # compute average CLIP score
    return total_score / count if count > 0 else 0.0
