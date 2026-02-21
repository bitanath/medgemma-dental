#!/usr/bin/env python3
"""
Convert dental dataset to tooth + problem tooth detection dataset.
Each record teeth bounding contains all boxes in one image.
"""

import json
import os

INPUT_JSON = "dataset_all/dataset.json"
OUTPUT_JSONL = "dataset_all/tooth_problem_dataset.jsonl"


def convert_box_to_paligemma_tokens(y1, x1, y2, x2):
    """Convert bbox (0-1023) to PaliGemma 2 format with zero-padding."""
    return f"<loc{y1:04d}><loc{x1:04d}><loc{y2:04d}><loc{x2:04d}>"


def process_dataset():
    """Process dataset and create JSONL with tooth/problem tooth labels."""
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
    
    prompt = "detect tooth; detect problem tooth;"
    
    with open(OUTPUT_JSONL, 'w') as f_out:
        for item in data:
            objects = item['objects']
            
            target_parts = []
            
            for obj in objects:
                # Get bbox coords
                y1, x1, y2, x2 = obj['y1'], obj['x1'], obj['y2'], obj['x2']
                
                # Convert to PaliGemma token format
                box_tokens = convert_box_to_paligemma_tokens(y1, x1, y2, x2)
                
                # Determine label based on treatment
                treatment = obj.get('treatment', 'none')
                if treatment == 'none':
                    label = 'tooth'
                else:
                    label = 'problem tooth'
                
                target_parts.append(f"{box_tokens} {label}")
            
            # Create sample
            sample = {
                "image": item['file'],
                "prompt": prompt,
                "target": "; ".join(target_parts),
                "num_objects": len(objects)
            }
            
            f_out.write(json.dumps(sample) + '\n')
    
    print(f"Created {OUTPUT_JSONL} with {len(data)} samples")
    
    # Print some stats
    problem_count = 0
    healthy_count = 0
    for item in data:
        for obj in item['objects']:
            if obj.get('treatment') == 'none':
                healthy_count += 1
            else:
                problem_count += 1
    
    print(f"Total healthy teeth: {healthy_count}")
    print(f"Total problem teeth: {problem_count}")


if __name__ == "__main__":
    process_dataset()
