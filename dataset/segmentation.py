import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Make sure the GroundingDINO folder is in sys.path for imports
import sys
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import GroundingDINO.groundingdino.util.box_ops as box_ops
from GroundingDINO.groundingdino.util.inference import predict
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.models import build_model

from segment_anything import SamPredictor, build_sam

def load_groundingdino_model(ckpt_path, config_path, device="cpu"):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    print(f"Loaded GroundingDINO model from {ckpt_path}")
    return model

# --------------- CONFIG ---------------
GROUNDING_CKPT = "groundingdino_swinb_cogcoor.pth"
GROUNDING_CFG = "imseg/GroundingDINO_SwinB.cfg.py"
SAM_CKPT = "imseg/sam_vit_h_4b8939.pth"

INPUT_DIR = "dataset/row_images"
OUTPUT_DIR = "dataset/segmented_images"
# --------------------------------------


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load models
grounding_model = load_groundingdino_model(GROUNDING_CKPT, GROUNDING_CFG, device)
sam_predictor = SamPredictor(build_sam(checkpoint=SAM_CKPT).to(device))

# Transform for GroundingDINO
transform = T.Compose([T.ToTensor()])

# Ensure base output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Iterate over each subfolder (each Pokémon)
for pokemon_name in os.listdir(INPUT_DIR):
    sub_input_folder = os.path.join(INPUT_DIR, pokemon_name)
    if not os.path.isdir(sub_input_folder):
        continue

    sub_output_folder = os.path.join(OUTPUT_DIR, pokemon_name)
    os.makedirs(sub_output_folder, exist_ok=True)

    print(f"\n Processing Pokémon: {pokemon_name}")
    text_prompt = f"pokemon character"

    for filename in os.listdir(sub_input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(sub_input_folder, filename)
        image_pil = Image.open(image_path)

        # If transparency exists, save as-is (segmentation NOT needed)
        if image_pil.mode in ("RGBA", "LA"):
            print(f"{filename} already has transparency, saving as-is.")
            save_path = os.path.join(sub_output_folder, os.path.splitext(filename)[0] + ".png")
            image_pil.save(save_path)
            continue

        image_pil = image_pil.convert("RGB")
        image_np = np.array(image_pil)
        image_tensor = transform(image_pil).to(device)

        try:
            boxes, logits, phrases = predict(
                model=grounding_model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=0.1,
                text_threshold=0.1
            )
        except RuntimeError as e:
            print(f" Skipping {filename} due to prediction error: {e}")
            continue

        if boxes.shape[0] == 0:
            print(f"No objects detected in {filename}")
            continue

        # Sort boxes by confidence score (logits)
        top_k = min(5, boxes.shape[0])
        sorted_indices = logits.argsort(descending=True)
        selected_boxes = boxes[sorted_indices[:top_k]]

        sam_predictor.set_image(image_np)
        H, W, _ = image_np.shape

        for i, box in enumerate(selected_boxes):
            box_xyxy = box_ops.box_cxcywh_to_xyxy(box.unsqueeze(0)) * torch.Tensor([W, H, W, H])
            transformed_box = sam_predictor.transform.apply_boxes_torch(box_xyxy.to(device), image_np.shape[:2])

            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_box,
                multimask_output=False
            )

            mask = masks[0][0].cpu().numpy()

            rgba = np.zeros((H, W, 4), dtype=np.uint8)
            rgba[..., :3] = image_np
            rgba[..., 3] = (mask * 255).astype(np.uint8)

            base_name = os.path.splitext(filename)[0]
            save_name = f"{base_name}_option_{i+1}.png"
            save_path = os.path.join(sub_output_folder, save_name)
            Image.fromarray(rgba).save(save_path)
            print(f" Saved segmented image: {save_path}")

print("\n All Pokémon processed.")