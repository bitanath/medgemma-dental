#!/usr/bin/env python3
"""
Script to crop individual teeth from dental images based on bounding boxes.
Creates a new dataset with square padded, 448x448 cropped images.
"""

import json
import os
from PIL import Image
from pathlib import Path


def crop_and_resize(image_path, bbox, output_path, target_size=448, expand_ratio=0.2):
    """
    Crop an image using the bounding box coordinates, pad to square with black,
    and resize to target size.
    
    Args:
        image_path: Path to the source image
        bbox: List of [x1, y1, x2, y2] coordinates
        output_path: Path to save the cropped image
        target_size: Target size for the square output (default 448)
        expand_ratio: Ratio to expand the bounding box (default 0.2 = 20%)
    """
    with Image.open(image_path) as img:
        x1, y1, x2, y2 = bbox
        
        # Expand bounding box by 20% in each direction
        width, height = img.size
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Calculate expansion
        x1 = int(x1 - expand_ratio * bbox_width)
        x2 = int(x2 + expand_ratio * bbox_width)
        y1 = int(y1 - expand_ratio * bbox_height)
        y2 = int(y2 + expand_ratio * bbox_height)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # Crop the image
        cropped = img.crop((x1, y1, x2, y2))
        
        # Calculate dimensions for square padding
        crop_width, crop_height = cropped.size
        max_dim = max(crop_width, crop_height)
        
        # Create a new black square image
        square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
        
        # Calculate position to center the crop
        paste_x = (max_dim - crop_width) // 2
        paste_y = (max_dim - crop_height) // 2
        
        # Paste the cropped image onto the black square
        square_img.paste(cropped, (paste_x, paste_y))
        
        # Resize to target size
        resized = square_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Save
        resized.save(output_path, quality=95)


def process_dataset(input_jsonl, output_dir, dataset_dir):
    """
    Process the dataset: crop images and create new metadata.
    
    Args:
        input_jsonl: Path to input dataset.jsonl
        output_dir: Path to output directory (individual/)
        dataset_dir: Path to dataset directory containing images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    output_jsonl = os.path.join(output_dir, 'dataset.jsonl')
    
    line_num = 0
    with open(input_jsonl, 'r') as infile, open(output_jsonl, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            data = json.loads(line.strip())
            
            # Extract fields
            image_name = data['image']
            tooth = data['tooth']
            diagnosis = data['diagnosis']
            treatment = data['treatment']
            bbox = data['bbox']
            
            # Create unique filename for cropped image
            base_name = Path(image_name).stem
            crop_filename = f"{base_name}_tooth_{line_num:05d}.jpg"
            output_image_path = os.path.join(output_dir, crop_filename)
            
            # Crop the image
            input_image_path = os.path.join(dataset_dir, image_name)
            if os.path.exists(input_image_path):
                crop_and_resize(input_image_path, bbox, output_image_path)
                
                # Create output metadata with same keys
                output_data = {
                    'tooth': tooth,
                    'treatment': treatment,
                    'diagnosis': diagnosis,
                    'filename': crop_filename
                }
                
                # Write to output jsonl
                outfile.write(json.dumps(output_data) + '\n')
                
                if line_num % 100 == 0:
                    print(f"Processed {line_num} images...")
            else:
                print(f"Warning: Image not found: {input_image_path}")
    
    print(f"\nComplete! Processed {line_num} entries.")
    print(f"Cropped images saved to: {output_dir}")
    print(f"Metadata saved to: {output_jsonl}")


if __name__ == '__main__':
    # Configuration
    INPUT_JSONL = 'dataset/dataset.jsonl'
    OUTPUT_DIR = 'individual'
    DATASET_DIR = 'dataset'
    
    process_dataset(INPUT_JSONL, OUTPUT_DIR, DATASET_DIR)
