import os
import torch
import gradio as gr
import spaces
from PIL import Image
from transformers import pipeline

torch.set_grad_enabled(False)

MODEL_NAME = "google/medgemma-1.5-4b-it"

pipe = None
DEVICE = "cpu"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_models():
    global pipe, DEVICE

    hf_token = os.environ.get("HF_TOKEN")
    DEVICE = get_device()
    print(f"Using device: {DEVICE}")

    print("Loading MedGemma model...")
    pipe = pipeline(
        "image-text-to-text",
        model=MODEL_NAME,
        dtype=torch.bfloat16,
        device=DEVICE,
        token=hf_token,
    )
    pipe.model.generation_config.do_sample = False
    pipe.model.generation_config.pad_token_id = (
        pipe.processor.tokenizer.eos_token_id
    )
    pipe.processor.tokenizer.padding_side = "left"

    print("Model loaded successfully!")


@spaces.GPU(duration=120)
def diagnose(image_path):
    if image_path is None:
        return "Please upload an image first."

    image = Image.open(image_path).convert("RGB")

    prompt_text = "Analyze this dental radiograph and provide findings."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    with torch.no_grad():
        output = pipe(
            text=messages,
            max_new_tokens=512,
            return_full_text=False,
        )

    diagnosis = output[0]["generated_text"]
    return diagnosis


def create_interface():
    with gr.Blocks(title="Dental Diagnosis") as demo:
        gr.Markdown("# 🦷 Dental Diagnosis with MedGemma")
        gr.Markdown(
            "Upload a dental X-ray image (OPG or IOPAR) and get AI-powered diagnosis."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="filepath", label="Upload Dental X-ray")

                gr.Examples(
                    examples=[
                        ["https://raw.githubusercontent.com/bitanath/medgemma-dental/main/example_xrays/extraction_img5.jpg"],
                        ["https://raw.githubusercontent.com/bitanath/medgemma-dental/main/example_xrays/panoramic_img109.jpg"],
                        ["https://raw.githubusercontent.com/bitanath/medgemma-dental/main/example_xrays/rct_img29.jpg"],
                    ],
                    inputs=[input_image],
                )

                diagnose_btn = gr.Button("🔍 Diagnose", variant="primary")

            with gr.Column(scale=1):
                diagnosis_result = gr.Textbox(
                    label="Diagnosis Result", lines=15, interactive=False
                )

        diagnose_btn.click(
            fn=diagnose,
            inputs=[input_image],
            outputs=[diagnosis_result],
        )

    return demo


if __name__ == "__main__":
    load_models()
    demo = create_interface()
    demo.launch(share=False)
