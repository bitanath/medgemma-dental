#!/usr/bin/env python3
"""
Convert dental dataset to MedGemma training format (JSONL).

This script reads dataset.json and creates a JSONL file with:
- Filtered entries (no 'none' for diagnosis or treatment)
- Human-like text descriptions combining treatment and diagnosis
- Bounding box information
- Compact JSON summary
- Type field (OPG/Periapical)

Usage:
    python convert_to_medgemma.py
"""

import json
import random
from pathlib import Path

# Configuration
INPUT_FILE = Path("dataset/dataset.json")
OUTPUT_FILE = Path("dataset/dataset.jsonl")

# Impersonal, clinical templates for combining treatment and diagnosis with type
TEXT_TEMPLATES = [
    "On the {type} image, the recommended treatment is {treatment} due to {diagnosis}.",
    "This {type} shows a condition that calls for {treatment} because {diagnosis}.",
    "Clinical findings on this {type} suggest {treatment} as {diagnosis}.",
    "The {type} indicates an appropriate intervention of {treatment} given that {diagnosis}.",
    "Treatment plan based on this {type} includes {treatment} as the diagnosis indicates {diagnosis}.",
    "Based on clinical evidence from this {type} of {diagnosis}, {treatment} is advised.",
    "The {type} reveals {diagnosis}, necessitating {treatment}.",
    "Considering the findings on this {type} of {diagnosis}, the best course of action is {treatment}.",
    "The {type} indicates treatment of {treatment} in response to {diagnosis}.",
    "This {type} demonstrates {diagnosis}, requiring {treatment}.",
]


def get_image_type(filename: str) -> str:
    """Determine image type based on filename."""
    if "panoramic" in filename.lower():
        return "OPG"
    else:
        return "Periapical"


def format_diagnosis_text(treatment: str, diagnosis: str, image_type: str) -> str:
    """Create an impersonal clinical text description from treatment, diagnosis, and type."""
    # Clean up the strings
    treatment = treatment.strip()
    diagnosis = diagnosis.strip()
    
    # Remove trailing punctuation from diagnosis if present
    if diagnosis.endswith(('.', '!', '?')):
        diagnosis = diagnosis[:-1]
    
    # Lowercase the first letter of diagnosis for mid-sentence insertion
    if diagnosis and diagnosis[0].isupper():
        diagnosis = diagnosis[0].lower() + diagnosis[1:]
    
    # Randomly select a template
    template = random.choice(TEXT_TEMPLATES)
    return template.format(treatment=treatment, diagnosis=diagnosis, type=image_type)


def convert_dataset():
    """Main conversion function."""
    print(f"Reading dataset from: {INPUT_FILE}")
    
    # Load the dataset
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} image entries")
    
    # Process and filter entries
    output_entries = []
    skipped_count = 0
    
    for image_entry in data:
        image_path = image_entry["file"]
        
        # Determine image type
        image_type = get_image_type(image_path)
        
        objects = image_entry.get("objects", [])
        
        for obj in objects:
            tooth = obj.get("tooth", "").strip()
            treatment = obj.get("treatment", "").strip().lower()
            diagnosis = obj.get("diagnosis", "").strip()
            
            # Filter: skip if treatment or diagnosis is "none" or empty
            if treatment in ["none", "", "null"] or diagnosis in ["none", "", "null"]:
                skipped_count += 1
                continue
            
            # Extract bounding box
            bbox = [obj["x1"], obj["y1"], obj["x2"], obj["y2"]]
            
            # Create human-like text summary with type
            text_summary = format_diagnosis_text(treatment, diagnosis, image_type)
            
            # Create compact JSON summary with type
            summary = {
                "tooth": tooth,
                "treatment": treatment,
                "diagnosis": diagnosis,  # Original diagnosis
                "type": image_type,  # Image type: OPG or Periapical
                "bbox": bbox
            }
            
            # Create output entry
            entry = {
                "image": image_path,
                "type": image_type,  # Image type: OPG or Periapical
                "tooth": tooth,
                "treatment": treatment,
                "diagnosis": diagnosis,  # Original diagnosis unchanged
                "bbox": bbox,
                "text_summary": text_summary,  # Human-readable combined text with type
                "json_summary": json.dumps(summary, separators=(',', ':'))
            }
            
            output_entries.append(entry)
    
    print(f"Filtered {skipped_count} entries with 'none' treatment/diagnosis")
    print(f"Writing {len(output_entries)} valid entries to: {OUTPUT_FILE}")
    
    # Write JSONL file
    with open(OUTPUT_FILE, 'w') as f:
        for entry in output_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Successfully created {OUTPUT_FILE}")
    print(f"\nSample entries:")
    for i, entry in enumerate(output_entries[:3]):
        print(f"\n--- Entry {i+1} ---")
        print(f"Image: {entry['image']}")
        print(f"Type: {entry['type']}")
        print(f"Tooth: {entry['tooth']}")
        print(f"Treatment: {entry['treatment']}")
        print(f"Diagnosis: {entry['diagnosis']}")
        print(f"BBox: {entry['bbox']}")
        print(f"Text Summary: {entry['text_summary']}")
        print(f"Summary: {entry['json_summary']}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    convert_dataset()
