import re
import os
import torch
import gradio as gr
import spaces
from PIL import Image, ImageDraw
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    pipeline,
    AutoImageProcessor,
    AutoModelForImageClassification,
)

torch.set_grad_enabled(False)

DETECTION_MODEL_NAME = "justacoderwhocodes/paligemma-dental-bounding-boxes"
DIAGNOSIS_MODEL_NAME = "justacoderwhocodes/medgemma-dental-diagnosis-finetune"
PROCESSOR_NAME = "google/paligemma2-3b-pt-448"
TREATMENT_MODEL_NAME = "justacoderwhocodes/dental-iopar-binary-classifier"

COLORS = {
    "molar": "purple",
    "premolar": "blue",
    "canine": "green",
    "incisor": "orange",
    "unknown": "grey",
}

detection_model = None
detection_processor = None
diagnosis_pipe = None
treatment_model = None
treatment_processor = None
DEVICE = "cpu"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_models():
    global detection_model, detection_processor, diagnosis_pipe, treatment_model, treatment_processor, DEVICE

    hf_token = os.environ.get("HF_TOKEN")
    DEVICE = get_device()
    print(f"Using device: {DEVICE}")

    print("Loading detection model (PaliGemma)...")
    detection_model = PaliGemmaForConditionalGeneration.from_pretrained(
        DETECTION_MODEL_NAME, dtype=torch.bfloat16, token=hf_token
    )
    detection_model = detection_model.to(DEVICE)
    detection_model = detection_model.eval()

    detection_processor = PaliGemmaProcessor.from_pretrained(
        PROCESSOR_NAME, token=hf_token
    )

    print("Loading diagnosis model (MedGemma)...")
    diagnosis_pipe = pipeline(
        "image-text-to-text",
        model=DIAGNOSIS_MODEL_NAME,
        dtype=torch.bfloat16,
        device=DEVICE,
        token=hf_token,
    )
    diagnosis_pipe.model.generation_config.do_sample = False
    diagnosis_pipe.model.generation_config.pad_token_id = (
        diagnosis_pipe.processor.tokenizer.eos_token_id
    )
    diagnosis_pipe.processor.tokenizer.padding_side = "left"

    print("Loading treatment classifier...")
    treatment_processor = AutoImageProcessor.from_pretrained(TREATMENT_MODEL_NAME, token=hf_token)
    treatment_model = AutoModelForImageClassification.from_pretrained(
        TREATMENT_MODEL_NAME,
        ignore_mismatched_sizes=True,
        token=hf_token,
    )
    treatment_model = treatment_model.to(DEVICE)
    treatment_model = treatment_model.eval()

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
        
        is_treatment = det.get("needs_treatment", False)
        
        tooth_type = label.lower().replace(" treatment", "")
        
        if is_treatment:
            color = "red"
            width = 4
        else:
            color = COLORS.get(tooth_type, "grey")
            width = 3
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw.text((x1, y1 - 12), f"{det['index'] + 1}: {tooth_type}", fill=color)

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


def square_pad_and_resize(image, target_size):
    width, height = image.size
    
    max_dim = max(width, height)
    
    new_image = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    new_image = new_image.resize((target_size, target_size), Image.LANCZOS)
    
    return new_image


@spaces.GPU(duration=55)
def classify_treatment(image, bbox):
    cropped = crop_bbox(image, bbox, expand_ratio=0.2)
    cropped = square_pad_and_resize(cropped, target_size=448)
    
    inputs = treatment_processor(images=cropped, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = treatment_model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
    
    label = treatment_model.config.id2label[pred]
    return label


@spaces.GPU(duration=55)
def detect_teeth(image_path):
    if image_path is None:
        return gr.update(visible=False, value=None), [], "Please upload an image first."

    image = Image.open(image_path).convert("RGB")
    image = square_pad_and_resize(image, target_size=1024)

    prompt = "<image><bos>detect canine; detect incisor; detect molar; detect premolar;"

    inputs = detection_processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = detection_model.generate(**inputs, max_new_tokens=512)

    result = detection_processor.decode(output[0], skip_special_tokens=False)

    detections = parse_bboxes(result, image.width, image.height)

    if not detections:
        return gr.update(visible=False, value=None), [], "No teeth detected in the image."

    treatment_count = 0
    total = len(detections)
    for i, det in enumerate(detections):
        treatment_label = classify_treatment(image, det["bbox"])
        if treatment_label == "treatment":
            det["needs_treatment"] = True
            treatment_count += 1
        else:
            det["needs_treatment"] = False
        
        progress_msg = f"Processing classification: {i+1}/{total} teeth..."

    annotated_image = draw_boxes(image, detections)

    detection_info = "\n".join(
        [f"[{d['index'] + 1}] {d['label'].replace(' treatment', '')} ({'treatment' if d['needs_treatment'] else 'no treatment'})" for d in detections]
    )

    treatment_msg = f"\n{treatment_count} tooth/teeth require treatment." if treatment_count > 0 else ""

    return gr.update(visible=True, value=annotated_image), detections, f"Detected {len(detections)} teeth. Classification complete.\n\n{detection_info}{treatment_msg}\n\nClick on a tooth to diagnose it."


def handle_click(image_path, detections, evt: gr.SelectData):
    if not detections or image_path is None:
        return gr.update(visible=False, value=None), "No detections available. Please run detection first.", False

    image = Image.open(image_path).convert("RGB")
    click_x, click_y = evt.index

    selected = None
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if x1 <= click_x <= x2 and y1 <= click_y <= y2:
            selected = det
            break

    if selected is None:
        return gr.update(visible=False, value=None), f"No tooth at click location ({click_x:.0f}, {click_y:.0f}). Click on a colored bounding box.", False

    cropped = crop_bbox(image, selected["bbox"], expand_ratio=0.2)
    
    cropped = square_pad_and_resize(cropped, target_size=448)

    treatment_status = "treatment" if selected.get("needs_treatment", False) else "no treatment"
    needs_treatment = selected.get("needs_treatment", False)
    return gr.update(visible=True, value=cropped), f"Selected: {selected['label'].replace(' treatment', '')} (Index {selected['index'] + 1}, {treatment_status})\nCropped region ready for diagnosis.", needs_treatment


@spaces.GPU(duration=55)
def diagnose_tooth(cropped_image, needs_treatment=False):
    if cropped_image is None:
        return "Please select a tooth first by clicking on a bounding box."
    
    if not needs_treatment:
        return "Diagnosis not provided as no treatment diagnosed"

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
            "Upload a dental X-ray image (OPG or IOPAR) → Detect teeth → Click on a tooth to diagnose"
        )
        gr.Markdown(
            "Note this space runs on `cpu` and might be slow. Appreciate your patience."
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

                detect_btn = gr.Button("🔍 Detect Teeth", variant="primary")

                status_text = gr.Textbox(
                    label="Status", lines=5, interactive=False
                )

                gr.Markdown(
            """
### Instructions:
1. Upload a dental X-ray image
2. Click "Detect Teeth" to run detection
3. Click on any detected tooth (colored bounding box) to select it
4. Click "Diagnose" to analyze the selected tooth

### Color Legend:
- 🔴 Red (thick): Treatment needed
- 🟣 Purple: Molar
- 🔵 Blue: Premolar
- 🟢 Green: Canine
- 🟠 Orange: Incisor
"""
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
        selected_treatment_state = gr.State(value=False)

        detect_btn.click(
            fn=detect_teeth,
            inputs=[input_image],
            outputs=[annotated_image, detections_state, status_text],
        )

        annotated_image.select(
            fn=handle_click,
            inputs=[input_image, detections_state],
            outputs=[cropped_image, status_text, selected_treatment_state],
        )

        diagnose_btn.click(
            fn=diagnose_tooth,
            inputs=[cropped_image, selected_treatment_state],
            outputs=[diagnosis_result],
        )

    return demo


if __name__ == "__main__":
    load_models()
    demo = create_interface()
    demo.launch(share=False)
