import os
import torch
from torch.utils.data import Dataset, DataLoader

from data.data_utils import load_image_fast
from data.data_utils import get_augmentation_transforms, get_basic_transforms


class PokemonDataset(Dataset):
    def __init__(self, df, folder_path, transform=None, text_mode="train", desc_prob=0.8):
        self.transform = transform
        self.text_mode = text_mode
        self.desc_prob = desc_prob

        if text_mode not in {"train", "test_intra", "test_novel"}:
            raise ValueError(f"Invalid text_mode: {text_mode}")

        self.samples = []
        text_by_name = df.set_index('name')[['description', 'train_paraphrase', 'test_paraphrase']].to_dict('index')

        for pokemon_name in os.listdir(folder_path):
            pokemon_path = os.path.join(folder_path, pokemon_name)
            if not os.path.isdir(pokemon_path) or pokemon_name not in text_by_name:
                if pokemon_name not in text_by_name:
                    print("Missing:", pokemon_name)
                continue

            texts = text_by_name[pokemon_name]

            for fname in os.listdir(pokemon_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    img_path = os.path.join(pokemon_path, fname)
                    self.samples.append((img_path, texts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, texts = self.samples[idx]
        image = load_image_fast(img_path)
        if self.transform:
            image = self.transform(image)

        if self.text_mode == "train":
            if torch.rand(1).item() < self.desc_prob:
                desc = texts['description']
            else:
                desc = texts['train_paraphrase']
        elif self.text_mode == "test_intra":
            desc = texts['test_paraphrase']
        elif self.text_mode == "test_novel":
            desc = texts['description']

        return image, desc


def create_dataloaders(df, images_dir="dataset/images", image_size=256, batch_size=16, num_workers=16):
    train_dataset = PokemonDataset(
        df=df,
        folder_path=os.path.join(images_dir, "train"),
        transform=get_augmentation_transforms(image_size=image_size),
        text_mode="train",
        desc_prob=0.8,
    )
    val_dataset = PokemonDataset(
        df=df,
        folder_path=os.path.join(images_dir, "validation"),
        transform=get_basic_transforms(image_size=image_size),
        text_mode="train",
        desc_prob=0.8,
    )
    test_intra_dataset = PokemonDataset(
        df=df,
        folder_path=os.path.join(images_dir, "test_intra"),
        transform=get_basic_transforms(image_size=image_size),
        text_mode="test_intra",
    )
    test_novel_dataset = PokemonDataset(
        df=df,
        folder_path=os.path.join(images_dir, "test_novel"),
        transform=get_basic_transforms(image_size=image_size),
        text_mode="test_novel",
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True),
        DataLoader(test_intra_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True),
        DataLoader(test_novel_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    )