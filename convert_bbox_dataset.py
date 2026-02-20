#!/usr/bin/env python3
"""
Convert dental dataset to single JSONL with hierarchical labels.
Creates bbox_dataset.jsonl with multiple target fields.
"""

import json
import os
from collections import Counter

INPUT_JSON = "dataset_all/dataset.json"
OUTPUT_JSONL = "bbox_dataset.jsonl"
IMAGE_BASE_PATH = "dataset_all"

# Hierarchical label mapping
TOOTH_HIERARCHY = {
    'central_incisor': {'group': 'incisor', 'fallback': 'tooth'},
    'lateral_incisor': {'group': 'incisor', 'fallback': 'tooth'},
    'canine': {'group': 'canine', 'fallback': 'tooth'},
    'first_premolar': {'group': 'premolar', 'fallback': 'tooth'},
    'second_premolar': {'group': 'premolar', 'fallback': 'tooth'},
    'first_molar': {'group': 'molar', 'fallback': 'tooth'},
    'second_molar': {'group': 'molar', 'fallback': 'tooth'},
    'third_molar': {'group': 'molar', 'fallback': 'tooth'}
}


def convert_box_to_paligemma_tokens(x1, y1, x2, y2):
    """Convert bbox (0-1023) to PaliGemma 2 format with zero-padding."""
    return f"<loc{y1:04d}><loc{x1:04d}><loc{y2:04d}><loc{x2:04d}>"


def create_target_for_granularity(obj, granularity='fine'):
    """Create target string for specific granularity."""
    tooth_type = obj['tooth']
    box_tokens = convert_box_to_paligemma_tokens(
        obj['x1'], obj['y1'], obj['x2'], obj['y2']
    )
    
    # Handle unknown tooth types
    if tooth_type not in TOOTH_HIERARCHY:
        label = 'tooth'  # Fallback for unknown
    elif granularity == 'fine':
        label = tooth_type.replace('_', ' ')
    elif granularity == 'group':
        label = TOOTH_HIERARCHY[tooth_type]['group']
    else:  # fallback
        label = 'tooth'
    
    # Add space between box tokens and label
    return f"{box_tokens} {label}"


def create_prompt(objects):
    """Create prompt listing unique tooth types present in image."""
    # Get unique tooth types
    tooth_types = set(['canine','incisor','premolar','molar'])
    # tooth_types = set([k.replace("_"," ") for k in list(TOOTH_HIERARCHY.keys())])
    
    # Create prompt: "detect canine; detect incisor; detect premolar; detect molar;"
    prompt_parts = [f"detect {t}" for t in sorted(tooth_types)]
    return "; ".join(prompt_parts) + ";"


def process_dataset():
    """Process dataset and create JSONL with all label granularities."""
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
    
    with open(OUTPUT_JSONL, 'w') as f_out:
        for item in data:
            objects = item['objects']
            
            # Create targets for each granularity (semicolon-separated)
            fine_targets = [create_target_for_granularity(obj, 'fine') for obj in objects]
            group_targets = [create_target_for_granularity(obj, 'group') for obj in objects]
            fallback_targets = [create_target_for_granularity(obj, 'fallback') for obj in objects]
            
            # Create prompt based on tooth types present
            prompt = create_prompt(objects)
            
            # Create single sample with all fields
            sample = {
                "image": os.path.join(IMAGE_BASE_PATH, item['file']).replace("dataset/",""),
                "prompt": prompt,
                "target": "; ".join(group_targets),  # Default: fine-grained (8 classes)
                "target_group": "; ".join(group_targets),  # 4 classes
                "target_fallback": "; ".join(fallback_targets),  # 1 class
                "num_objects": len(objects)
            }
            
            f_out.write(json.dumps(sample) + '\n')
    
    print(f"Created {OUTPUT_JSONL} with {len(data)} samples")


if __name__ == "__main__":
    process_dataset()
