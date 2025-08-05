import torch
import os

from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from tqdm import tqdm

from training.utils import find_latest_checkpoint, load_checkpoint, save_checkpoint, generate_and_save_images
from training.metrics import compute_fid

from torch.amp import GradScaler, autocast


################ TRAIN STEP ################
def train_step(real_imgs, texts, device, generator, discriminator, text_encoder, 
               optimizerG, optimizerD,
               scalerG, scalerD):
    batch_size = real_imgs.size(0)

    # -----------------------------
    # Discriminator forward + backward
    # -----------------------------
    noise = torch.randn(batch_size, generator.z_dim).to(device)

    with autocast(device_type="cuda", dtype=torch.float16):
        per_token_emb, global_emb = text_encoder(texts)
        fake_imgs = generator(noise, per_token_emb, global_emb)

    # Needed to compute MA-GP
    global_emb_for_gp = global_emb.to(torch.float32).requires_grad_()
    real_imgs = real_imgs.to(device).requires_grad_()

    # Perform the discriminator forward pass for REAL images (and MA-GP) OUTSIDE autocast.
    out_real = discriminator(real_imgs, global_emb_for_gp) # real_imgs and global_emb_for_gp are float32 -> out_real still in float32

    # Back to autocast for other discriminator passes
    with autocast(device_type="cuda", dtype=torch.float16):
        out_fake = discriminator(fake_imgs.detach(), global_emb)

        mismatched_emb = torch.roll(global_emb, shifts=1, dims=0)
        out_mis = discriminator(real_imgs.detach(), mismatched_emb)

        # Hinge losses
        loss_real = torch.relu(1.0 - out_real).mean()
        loss_fake = torch.relu(1.0 + out_fake).mean()
        loss_mis = torch.relu(1.0 + out_mis).mean()

    # MA-GP (inputs and outputs are float32, ensures stability)
    grads = torch.autograd.grad(
        outputs=out_real, # This is float32
        inputs=(real_imgs, global_emb_for_gp), # These are float32
        grad_outputs=torch.ones_like(out_real), # This will be float32
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )
    grad_img = grads[0].view(batch_size, -1)
    grad_emb = grads[1].view(batch_size, -1)
    grad = torch.cat([grad_img, grad_emb], dim=1)
    grad_l2_norm = torch.sqrt(torch.sum(grad ** 2, dim=1) + 1e-8)
    loss_magp = 1.5 * torch.mean(grad_l2_norm ** 6)

    # loss_D calculation
    loss_D = loss_real + 0.5 * (loss_fake + loss_mis) + loss_magp

    optimizerD.zero_grad()
    scalerD.scale(loss_D).backward()
    scalerD.step(optimizerD)
    scalerD.update()

    # -----------------------------
    # Generator forward + backward
    # -----------------------------
    with autocast(device_type="cuda", dtype=torch.float16):
        # Re-encode text for G
        per_token_emb, global_emb = text_encoder(texts)
        per_token_emb = per_token_emb.to(device)
        global_emb = global_emb.to(device)

        fake_imgs = generator(noise, per_token_emb, global_emb)
        out_fake_G = discriminator(fake_imgs, global_emb)
        loss_G = -out_fake_G.mean()

    optimizerG.zero_grad()
    scalerG.scale(loss_G).backward()
    scalerG.step(optimizerG)
    scalerG.update()

    return loss_real.item(), loss_fake.item(), loss_mis.item(), loss_magp.item(), loss_D.item(), loss_G.item()

@torch.no_grad()
def valid_step(generator, text_encoder, val_loader, fixed_z, writer, epoch, device, nrow=4, tag="Validation"):
    generator.eval()
    text_encoder.eval()

    val_iter = iter(val_loader)
    real_imgs, texts = next(val_iter)
    batch_size = real_imgs.size(0)

    real_imgs = real_imgs.to(device)
    z = fixed_z[:batch_size].to(device)

    with autocast(device_type="cuda", dtype=torch.float16):
        per_token_emb, global_emb = text_encoder(texts)
        fake_imgs = generator(z, per_token_emb, global_emb)

    # Normalize images to [0, 1] for display and move to CPU
    real_imgs_cpu_normalized = ((real_imgs - real_imgs.min()) / (real_imgs.max() - real_imgs.min() + 1e-8)).cpu() # Added epsilon for safety
    fake_imgs_cpu_normalized = ((fake_imgs - fake_imgs.min()) / (fake_imgs.max() - fake_imgs.min() + 1e-8)).cpu() # Added epsilon for safety

    pil_real_imgs = [to_pil_image(img_tensor) for img_tensor in real_imgs_cpu_normalized]
    pil_fake_imgs = [to_pil_image(img_tensor) for img_tensor in fake_imgs_cpu_normalized]

    font = ImageFont.load_default()

    # Function to add a single caption to a combined image
    def add_combined_caption(real_img_pil, fake_img_pil, caption_text, font_obj):
        img_W, img_H = real_img_pil.size

        # Create a dummy draw object to estimate text height (using the font)
        dummy_img = Image.new("RGB", (1,1))
        dummy_draw = ImageDraw.Draw(dummy_img)

        # Estimate line height once using a common character, to maintain consistent line spacing
        try:
            # textbbox returns (left, top, right, bottom)
            # Use a typical character like 'H' or 'g' to get a representative line height
            _, _, _, baseline_height = dummy_draw.textbbox((0, 0), "Hg", font=font_obj) 
        except AttributeError: # Fallback for older Pillow versions
            _, baseline_height = dummy_draw.textsize("Hg", font=font_obj)
        if baseline_height == 0: # Ensure it's not zero if font is problematic
            baseline_height = 16 # Fallback to a reasonable default

        # Text wrapping logic
        combined_width = img_W * 2 # Width for real + fake image
        wrapped_lines = []
        current_line = []
        words = caption_text.split(' ')
        max_lines_display = 3 # Max lines for the combined caption

        for word in words:
            test_line = ' '.join(current_line + [word])
            try:
                # Use textbbox to measure text width
                left, top, right, bottom = dummy_draw.textbbox((0,0), test_line, font=font_obj)
                text_w = right - left
            except AttributeError:
                # Fallback to textsize for older Pillow versions
                text_w, _ = dummy_draw.textsize(test_line, font=font_obj)
            
            if text_w <= combined_width - 20 and len(wrapped_lines) < max_lines_display: # 20 for padding
                current_line.append(word)
            else:
                wrapped_lines.append(' '.join(current_line))
                current_line = [word]
                if len(wrapped_lines) >= max_lines_display:
                    wrapped_lines[-1] += "..."
                    break
        if current_line and len(wrapped_lines) < max_lines_display:
            wrapped_lines.append(' '.join(current_line))

        # Calculate required canvas height for wrapped text
        total_text_height = sum([baseline_height for _ in wrapped_lines]) # Use estimated line height
        padding_text_vertical = 10 # Padding above and below text
        canvas_height = img_H + total_text_height + padding_text_vertical

        canvas = Image.new("RGB", (combined_width, canvas_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Paste images
        canvas.paste(real_img_pil, (0, 0))
        canvas.paste(fake_img_pil, (img_W, 0))

        # Draw wrapped caption below the images
        y_offset = img_H + padding_text_vertical // 2
        for line in wrapped_lines:
            try:
                left, top, right, bottom = dummy_draw.textbbox((0,0), line, font=font_obj)
                text_w = right - left
            except AttributeError:
                text_w, _ = dummy_draw.textsize(line, font=font_obj)

            x_center = (combined_width - text_w) / 2 # Center the text
            draw.text((x_center, y_offset), line, font=font_obj, fill=(0, 0, 0))
            y_offset += baseline_height # Move to next line using estimated line height

        return canvas

    # Iterate and combine for each pair
    combined_pairs_with_text = []
    for i in range(batch_size):
        combined_pairs_with_text.append(add_combined_caption(pil_real_imgs[i], pil_fake_imgs[i], texts[i], font)) # Pass font

    # Arrange images in a grid
    num_grid_rows = (len(combined_pairs_with_text) + nrow - 1) // nrow

    if not combined_pairs_with_text: # Handle empty list case
        print("Warning: No images to display in valid_step.")
        return

    grid_rows = []
    for r_idx in range(num_grid_rows):
        start_idx = r_idx * nrow
        end_idx = min((r_idx + 1) * nrow, len(combined_pairs_with_text))
        
        current_row_images = combined_pairs_with_text[start_idx:end_idx]
        
        if not current_row_images: continue

        row_width = sum(img.width for img in current_row_images) # Sum widths for dynamic row width
        row_height = current_row_images[0].height # All images in a row should have same height
        
        current_grid_row = Image.new("RGB", (row_width, row_height), (255, 255, 255))
        
        x_offset_in_row = 0
        for img_in_row in current_row_images:
            current_grid_row.paste(img_in_row, (x_offset_in_row, 0))
            x_offset_in_row += img_in_row.width
        grid_rows.append(current_grid_row)

    # Stack all rows vertically
    if not grid_rows:
        print("Error: Grid rows could not be formed.")
        return

    full_grid_width = grid_rows[0].width
    full_grid_height = sum(row.height for row in grid_rows)

    full_grid = Image.new("RGB", (full_grid_width, full_grid_height), (255, 255, 255))
    y_offset_in_grid = 0
    for row_img in grid_rows:
        full_grid.paste(row_img, (0, y_offset_in_grid))
        y_offset_in_grid += row_img.height

    # Convert back to tensor
    grid_tensor = torch.from_numpy(np.array(full_grid)).permute(2, 0, 1) / 255.0

    writer.add_image(f"{tag}_real_vs_fake", grid_tensor, epoch)

    generator.train()
    text_encoder.train()


################ TRAINING LOOP ################
def train_loop(num_epochs, train_loader, val_loader, generator, discriminator, text_encoder,
               optimizerG, optimizerD, fixed_z, device, checkpoint_dir, writer,
               real_val_folder, gen_val_folder, ckpt_interval=5, fid_interval=10,
               resume_checkpoint=False
               ):
    
    # Create GradScalers outside and pass them into train_step
    scalerG = GradScaler(device='cuda')
    scalerD = GradScaler(device='cuda')

    # Initialize or restore training state variables
    start_epoch = 0
    start_batch_idx = 0
    global_step = 0
    best_fid = float("inf")

    if resume_checkpoint:
        # Load checkpoint state, including optimizer states and global step
        if os.path.isdir(checkpoint_dir):
            resume_path = find_latest_checkpoint(checkpoint_dir)
            if resume_path is None:
                raise FileNotFoundError(f"No checkpoints found in directory {checkpoint_dir}")
        else:
            resume_path = resume_checkpoint

        # Unpack checkpoint_data
        start_epoch, fixed_z, best_fid, global_step = load_checkpoint(
            generator, discriminator, text_encoder,
            optimizerG, optimizerD, scalerG, scalerD, resume_path, device
        )

    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True)

        # If resuming, skip batches before start_batch_idx
        for i, (real_imgs, texts) in pbar:
            if epoch == start_epoch and i < start_batch_idx:
                # Skip batches that were already processed
                continue

            current_step = global_step

            loss_real_D, loss_fake_D, loss_miss_D, loss_magp_D, loss_D, loss_G = train_step(
                real_imgs=real_imgs, texts=texts, device=device, generator=generator, discriminator=discriminator,
                text_encoder=text_encoder, optimizerG=optimizerG, optimizerD=optimizerD,
                scalerG=scalerG, scalerD=scalerD,
            )

            pbar.set_postfix({
                "loss_real_D": f"{loss_real_D:.4f}",
                "loss_fake_D": f"{loss_fake_D:.4f}",
                "loss_miss_D": f"{loss_miss_D:.4f}",
                "loss_magp_D": f"{loss_magp_D:.4f}",
                "total_loss_D": f"{loss_D:.4f}",
                "loss_G": f"{loss_G:.4f}"
            })

            if i % 100 == 0:
                writer.add_scalar("Loss/D_real", loss_real_D, current_step)
                writer.add_scalar("Loss/D_fake", loss_fake_D, current_step)
                writer.add_scalar("Loss/D_mismatch", loss_miss_D, current_step)
                writer.add_scalar("Loss/D_magp", loss_magp_D, current_step)
                writer.add_scalar("Loss/D_total", loss_D, current_step)
                writer.add_scalar("Loss/G", loss_G, current_step)
                
            global_step += 1
        start_batch_idx = 0

        # Validation
        valid_step(generator, text_encoder, val_loader, fixed_z, writer, epoch, device)

        # Save checkpoint every ckpt_interval epochs
        if (epoch + 1) % ckpt_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "optimizerG": optimizerG.state_dict(),
                "optimizerD": optimizerD.state_dict(),
                "scalerG": scalerG.state_dict(),
                "scalerD": scalerD.state_dict(),
                "fixed_z": fixed_z,
                "fid_score": best_fid,
                "global_step": global_step,
            }, ckpt_path)

        # Compute FID every fid_interval epochs
        if (epoch + 1) % fid_interval == 0:
            gen_val_folder_epoch = os.path.join(gen_val_folder, f"valid_gen_epoch_{epoch+1}")
            generate_and_save_images(generator, text_encoder, val_loader, device, gen_val_folder_epoch, use_fixed_z=True, fixed_z=fixed_z, return_prompts=False)
            fid_score = compute_fid(real_val_folder, gen_val_folder_epoch)

            writer.add_scalar("Val_metrics/FID", fid_score, epoch + 1)
            print(f"Epoch {epoch+1}: FID score = {fid_score:.2f}")

            if fid_score < best_fid:
                best_fid = fid_score
                best_ckpt_path = os.path.join(checkpoint_dir, "best_fid_checkpoint.pth")
                save_checkpoint({
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "text_encoder": text_encoder.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                    "optimizerD": optimizerD.state_dict(),
                    "scalerG": scalerG.state_dict(),
                    "scalerD": scalerD.state_dict(),
                    "fixed_z": fixed_z,
                    "fid_score": fid_score,
                    "global_step": global_step,
                }, best_ckpt_path)
                print(f"New best FID checkpoint saved at epoch {epoch+1}")

    print(f"Training finished. Best FID: {best_fid:.2f}")
    writer.close()
