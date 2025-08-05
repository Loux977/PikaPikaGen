import os
import random

from PIL import Image

import os
import random
from PIL import Image

def split_dataset(dataset_root, output_root, train_ratio=0.9, val_ratio=0.03, test_ratio=0.07, seed=42):
    print("Performing split...")

    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Train, validation and test ratios must sum to 1.0")

    random.seed(seed)

    # Define split directories
    train_dir = os.path.join(output_root, 'train')
    val_dir = os.path.join(output_root, 'validation')
    test_dir = os.path.join(output_root, 'test')

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    pokemon_classes = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    total_images_processed = 0

    for pokemon_class in pokemon_classes:
        class_path = os.path.join(dataset_root, pokemon_class)

        # Prepare output subfolders
        for base_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(base_dir, pokemon_class), exist_ok=True)

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
        random.shuffle(images)

        num_images = len(images)
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)

        splits = {
            train_dir: images[:num_train],
            val_dir: images[num_train:num_train + num_val],
            test_dir: images[num_train + num_val:]
        }

        for split_dir, split_images in splits.items():
            for img_name in split_images:
                src_img_path = os.path.join(class_path, img_name)
                dest_img_name = os.path.splitext(img_name)[0] + ".jpg"
                dest_img_path = os.path.join(split_dir, pokemon_class, dest_img_name)

                try:
                    with Image.open(src_img_path) as img:
                        rgb_img = img.convert("RGB")
                        rgb_img.save(dest_img_path, format='JPEG', quality=95)
                except Exception as e:
                    print(f"Failed to process {src_img_path}: {e}")

        print(f"Processed class '{pokemon_class}': {num_images} images")
        total_images_processed += num_images

    print(f"\n Dataset split complete. Total images processed: {total_images_processed}")
    print(f"  Split dataset saved to: {output_root}")


def convert_rgba_dir_to_rgb_with_white_bg(input_dir, output_dir):
    """
    Converts all images in a directory tree (with subfolders) to RGB,
    replacing transparent backgrounds with white if present.
    
    Args:
        input_dir (str): Root directory with images (in subfolders)
        output_dir (str): Root output directory where converted images will be saved
    """
    for root, _, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        out_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(out_subdir, exist_ok=True)

        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue  # Skip non-image files

            in_path = os.path.join(root, file)
            out_path = os.path.join(out_subdir, os.path.splitext(file)[0] + ".png")

            try:
                img = Image.open(in_path)
                if img.mode == "RGBA":
                    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    img = Image.alpha_composite(background, img).convert("RGB")
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                img.save(out_path, format="PNG")
            except Exception as e:
                print(f"Failed to process {in_path}: {e}")
