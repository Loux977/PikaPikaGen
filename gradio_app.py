import os
import torch
from PIL import Image
from torchvision import transforms

import gradio as gr

from data.data_utils import set_all_seeds

from models.text_encoder import TextEncoder
from models.generator import Generator

@torch.no_grad()
def gradio_generate_image(prompt: str, use_fixed_seed: bool, seed_value: int) -> Image.Image:
    generator.eval()
    text_encoder.eval()

    effective_seed = seed_value if use_fixed_seed else None
    set_all_seeds(effective_seed)

    with torch.autocast(device_type=DEVICE.type):
        z = torch.randn(1, generator.z_dim).to(DEVICE)
        per_token_emb, global_emb = text_encoder(prompt)
        gen_img_tensor = generator(z, per_token_emb, global_emb)
        gen_img_tensor = (gen_img_tensor + 1.0) / 2.0
        gen_img_tensor = gen_img_tensor.squeeze(0)
        to_pil_image = transforms.ToPILImage()
        generated_pil_image = to_pil_image(gen_img_tensor)

    return generated_pil_image


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = "runs/experiment_1/checkpoints"

    print("Initializing models...")
    text_encoder = TextEncoder().to(DEVICE)
    generator = Generator().to(DEVICE)

    checkpoint_file = os.path.join(CHECKPOINT_DIR, "best_fid_checkpoint.pth")
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_file}")

    try:
        checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
        generator.load_state_dict(checkpoint["generator"])
        text_encoder.load_state_dict(checkpoint["text_encoder"])
        print("Models loaded successfully from checkpoint.")
    except Exception as e:
        print(f"Error loading models from checkpoint: {e}")
        exit()

    generator.eval()
    text_encoder.eval()

    # --- Components ---
    text_input = gr.Textbox(
        label="Text Description",
        placeholder="Enter a description for the image...",
        lines=2,
        value="It is a light-blue, penguin-like creature with thick downy feathers."
    )

    seed_checkbox = gr.Checkbox(
        label="Use Fixed Seed",
        value=True,
        interactive=True
    )

    seed_number = gr.Number(
        label="Seed Value",
        value=19,
        precision=0,
        minimum=0,
        maximum=2**32 - 1,
        interactive=True
    )

    image_output = gr.Image(
        label="Generated Image",
        type="pil"
    )

    # Create blocks UI
    with gr.Blocks() as demo:
        gr.Markdown("## Text-to-Image Pokémon Generator")
        with gr.Row():
            text_input.render()
        with gr.Row():
            seed_checkbox.render()
            seed_number.render()
        with gr.Row():
            generate_btn = gr.Button("Generate")
            clear_btn = gr.Button("Clear")

        with gr.Row():
            image_output.render()

        # Bind generation function
        generate_btn.click(
            fn=gradio_generate_image,
            inputs=[text_input, seed_checkbox, seed_number],
            outputs=image_output
        )

        # Custom clear behavior
        def reset_all():
            return "", True, 19, None

        clear_btn.click(
            fn=reset_all,
            inputs=[],
            outputs=[text_input, seed_checkbox, seed_number, image_output]
        )

    print("Launching Gradio app...")
    demo.launch(share=False, inbrowser=True)


