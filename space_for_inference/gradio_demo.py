import re
import torch
import gradio as gr
from PIL import Image, ImageDraw
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    pipeline,
)

torch.set_grad_enabled(False)

DETECTION_MODEL_NAME = "justacoderwhocodes/paligemma-dental-bounding-boxes"
DIAGNOSIS_MODEL_NAME = "justacoderwhocodes/medgemma-dental-diagnosis-finetune"
PROCESSOR_NAME = "google/paligemma2-3b-pt-448"

COLORS = {
    "molar": "red",
    "premolar": "blue",
    "canine": "green",
    "incisor": "orange",
    "unknown": "grey",
}

detection_model = None
detection_processor = None
diagnosis_pipe = None
DEVICE = None


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_models():
    global detection_model, detection_processor, diagnosis_pipe, DEVICE

    DEVICE = get_device()
    print(f"Using device: {DEVICE}")

    print("Loading detection model (PaliGemma)...")
    detection_model = PaliGemmaForConditionalGeneration.from_pretrained(
        DETECTION_MODEL_NAME, torch_dtype=torch.bfloat16
    )
    detection_model = detection_model.to(DEVICE)
    detection_model = detection_model.eval()

    detection_processor = PaliGemmaProcessor.from_pretrained(
        PROCESSOR_NAME
    )

    print("Loading diagnosis model (MedGemma)...")
    diagnosis_pipe = pipeline(
        "image-text-to-text",
        model=DIAGNOSIS_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device=DEVICE,
    )
    diagnosis_pipe.model.generation_config.do_sample = False
    diagnosis_pipe.model.generation_config.pad_token_id = (
        diagnosis_pipe.processor.tokenizer.eos_token_id
    )
    diagnosis_pipe.processor.tokenizer.padding_side = "left"

    print("Models loaded successfully!")


def parse_bboxes(model_output, img_width, img_height):
    pattern = r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([^;<]+)"
    matches = re.findall(pattern, model_output)

    detections = []
    for ymin, xmin, ymax, xmax, label in matches:
        ymin = int(ymin)
        xmin = int(xmin)
        ymax = int(ymax)
        xmax = int(xmax)

        x1 = xmin / 1023 * img_width
        y1 = ymin / 1023 * img_height
        x2 = xmax / 1023 * img_width
        y2 = ymax / 1023 * img_height

        label_clean = label.strip()
        detections.append(
            {"bbox": [x1, y1, x2, y2], "label": label_clean, "index": len(detections)}
        )

    return detections


def draw_boxes(image, detections):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        color = COLORS.get(label.lower(), "grey")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 12), f"{det['index']}: {label}", fill=color)

    return img


def crop_bbox(image, bbox, expand_ratio=0.2):
    x1, y1, x2, y2 = bbox
    width, height = image.size

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    x1 = int(x1 - expand_ratio * bbox_width)
    x2 = int(x2 + expand_ratio * bbox_width)
    y1 = int(y1 - expand_ratio * bbox_height)
    y2 = int(y2 + expand_ratio * bbox_height)

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    return image.crop((x1, y1, x2, y2))


def detect_teeth(image_path):
    if image_path is None:
        return gr.update(visible=False, value=None), [], "Please upload an image first."

    image = Image.open(image_path).convert("RGB")

    prompt = "<image><bos>detect canine; detect incisor; detect molar; detect premolar;"

    inputs = detection_processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = detection_model.generate(**inputs, max_new_tokens=512)

    result = detection_processor.decode(output[0], skip_special_tokens=False)

    detections = parse_bboxes(result, image.width, image.height)

    if not detections:
        return gr.update(visible=False, value=None), [], "No teeth detected in the image."

    annotated_image = draw_boxes(image, detections)

    detection_info = "\n".join(
        [f"[{d['index']}] {d['label']} at ({d['bbox'][0]:.0f}, {d['bbox'][1]:.0f}, {d['bbox'][2]:.0f}, {d['bbox'][3]:.0f})" for d in detections]
    )

    return gr.update(visible=True, value=annotated_image), detections, f"Detected {len(detections)} teeth:\n{detection_info}\n\nClick on a tooth to diagnose it."


def handle_click(image_path, detections, evt: gr.SelectData):
    if not detections or image_path is None:
        return gr.update(visible=False, value=None), "No detections available. Please run detection first."

    image = Image.open(image_path).convert("RGB")
    click_x, click_y = evt.index

    selected = None
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if x1 <= click_x <= x2 and y1 <= click_y <= y2:
            selected = det
            break

    if selected is None:
        return gr.update(visible=False, value=None), f"No tooth at click location ({click_x:.0f}, {click_y:.0f}). Click on a colored bounding box."

    cropped = crop_bbox(image, selected["bbox"])

    return gr.update(visible=True, value=cropped), f"Selected: {selected['label']} (Index {selected['index']})\nCropped region ready for diagnosis."


def diagnose_tooth(cropped_image):
    if cropped_image is None:
        return "Please select a tooth first by clicking on a bounding box."

    prompt_text = "Analyze this dental radiograph and provide findings."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": cropped_image},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    with torch.no_grad():
        output = diagnosis_pipe(
            text=messages,
            max_new_tokens=150,
            return_full_text=False,
        )

    diagnosis = output[0]["generated_text"]
    return diagnosis


def create_interface():
    with gr.Blocks(title="Dental Analysis Demo") as demo:
        gr.Markdown("# 🦷 Medgemma and MedSiglip based AI Dental Diagnosis ✨")
        gr.Markdown(
            "Upload a dental X-ray image → Detect teeth → Click on a tooth to diagnose"
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="filepath", label="Upload Dental X-ray")

                detect_btn = gr.Button("🔍 Detect Teeth", variant="primary")

                status_text = gr.Textbox(
                    label="Status", lines=5, interactive=False
                )

            with gr.Column(scale=1):
                annotated_image = gr.Image(
                    type="filepath", label="Detected Teeth (Click to select)", 
                    visible=False, interactive=False
                )

                cropped_image = gr.Image(
                    type="pil", label="Selected Tooth (Cropped)", 
                    visible=False, interactive=False
                )

                diagnose_btn = gr.Button("📋 Diagnose", variant="secondary")

                diagnosis_result = gr.Textbox(
                    label="Diagnosis Result", lines=5, interactive=False
                )

        detections_state = gr.State([])

        detect_btn.click(
            fn=detect_teeth,
            inputs=[input_image],
            outputs=[annotated_image, detections_state, status_text],
        )

        annotated_image.select(
            fn=handle_click,
            inputs=[input_image, detections_state],
            outputs=[cropped_image, status_text],
        )

        diagnose_btn.click(
            fn=diagnose_tooth,
            inputs=[cropped_image],
            outputs=[diagnosis_result],
        )

        gr.Markdown(
            """
        ### Instructions:
        1. Upload a dental X-ray image
        2. Click "Detect Teeth" to run detection
        3. Click on any detected tooth (colored bounding box) to select it
        4. Click "Diagnose" to analyze the selected tooth

        ### Color Legend:
        - 🔴 Red: Molar
        - 🔵 Blue: Premolar  
        - 🟢 Green: Canine
        - 🟠 Orange: Incisor
        """
        )

    return demo


if __name__ == "__main__":
    load_models()
    demo = create_interface()
    demo.launch(share=False)
