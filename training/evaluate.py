import torch
import os
from training.metrics import (
    compute_fid,
    compute_inception_score,
    compute_kid_score,
    compute_clip_score
)
from training.utils import generate_and_save_images


def evaluate_test_sets(generator, text_encoder, test_sets, device,
                       checkpoint_dir, use_fixed_z=False, results_file=None):
    """
    Evaluate generator on multiple test sets using saved checkpoint.

    Args:
        generator, text_encoder: models
        test_sets: dict mapping test name to tuple (loader, real_folder, gen_folder)
        device: torch device
        checkpoint_dir: where to load checkpoint from
        use_fixed_z: whether to use fixed z vectors from checkpoint
        results_file: path to output results file
    """

    # ---------------------------
    # Load checkpoint
    # ---------------------------
    checkpoint_file = os.path.join(checkpoint_dir, "best_fid_checkpoint.pth")
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_file}")
    
    checkpoint = torch.load(checkpoint_file, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    generator.eval()
    text_encoder.eval()

    # ---------------------------
    # Load fixed z from checkpoint if needed
    # ---------------------------
    fixed_z = checkpoint["fixed_z"] if use_fixed_z and "fixed_z" in checkpoint else None
    if use_fixed_z and fixed_z is not None:
        print("Loaded fixed z vectors from checkpoint.")
    elif use_fixed_z:
        raise ValueError("use_fixed_z=True but no fixed_z found in checkpoint.")
    else:
        print("Using random z vectors for evaluation.")

    # ---------------------------
    # Prepare output file
    # ---------------------------
    if results_file is None:
        raise ValueError("Must provide a path for `results_file` to save metrics.")

    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # ---------------------------
    # Evaluate each test set
    # ---------------------------
    with open(results_file, "w") as f:
        for name, (loader, real_folder, gen_folder) in test_sets.items():
            os.makedirs(gen_folder, exist_ok=True)
            print(f"\nEvaluating {name} set ...")
            
            # Generate images
            prompts = generate_and_save_images(
                generator=generator,
                text_encoder=text_encoder,
                dataloader=loader,
                device=device,
                save_folder=gen_folder,
                use_fixed_z=use_fixed_z,
                fixed_z=fixed_z,
                return_prompts=True
            )

            # Compute metrics
            fid = compute_fid(real_folder, gen_folder)
            is_mean, is_std = compute_inception_score(gen_folder)
            kid_mean, kid_std = compute_kid_score(real_folder, gen_folder)
            clip_score = compute_clip_score(gen_folder, prompts, device=device)

            # Print
            print(f"=== {name.capitalize()} Evaluation Results ===")
            print(f"{'FID':<20}: {fid:.3f}")
            print(f"{'Inception Score':<20}: mean = {is_mean:.3f}, std = {is_std:.5f}")
            print(f"{'KID':<20}: mean = {kid_mean:.3f}, std = {kid_std:.5f}")
            print(f"{'CLIP Score':<20}: {clip_score:.3f}")

            # Save to file
            f.write(f"=== {name.capitalize()} Evaluation Results ===\n")
            f.write(f"{'FID':<20}: {fid:.3f}\n")
            f.write(f"{'Inception Score':<20}: mean = {is_mean:.3f}, std = {is_std:.5f}\n")
            f.write(f"{'KID':<20}: mean = {kid_mean:.3f}, std = {kid_std:.5f}\n")
            f.write(f"{'CLIP Score':<20}: {clip_score:.3f}\n\n")

    print(f"\nAll test metrics saved to: {results_file}")
