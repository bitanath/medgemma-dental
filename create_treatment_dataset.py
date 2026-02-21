#!/usr/bin/env python3
"""
Script to create balanced treatment classification dataset from dataset_all/dataset.json
- Crops teeth using 1.2x bbox expansion
- Square pads with black, resizes to 448x448
- Balances none treatments to match other treatments (~1072 samples)
"""

import json
import os
import random
from PIL import Image
from pathlib import Path
from collections import defaultdict

random.seed(42)

def crop_and_resize(image_path, bbox, output_path, target_size=448, expand_ratio=1.2):
    """
    Crop an image using the bounding box coordinates, pad to square with black,
    and resize to target size.
    """
    with Image.open(image_path) as img:
        x1, y1, x2, y2 = bbox
        
        # Expand bounding box by expand_ratio in each direction (1.2 = 20%)
        width, height = img.size
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Calculate expansion
        x1 = int(x1 - (expand_ratio - 1) * bbox_width / 2)
        x2 = int(x2 + (expand_ratio - 1) * bbox_width / 2)
        y1 = int(y1 - (expand_ratio - 1) * bbox_height / 2)
        y2 = int(y2 + (expand_ratio - 1) * bbox_height / 2)
        
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

def main():
    # Load dataset
    print("Loading dataset...")
    with open('dataset_all/dataset.json') as f:
        data = json.load(f)
    
    print(f"Total images: {len(data)}")
    
    # Group objects by treatment
    treatment_groups = defaultdict(list)
    
    for item in data:
        image_name = item['file']
        image_path = os.path.join('dataset_all', image_name)
        
        for obj in item.get('objects', []):
            treatment = obj['treatment']
            bbox = [obj['x1'], obj['y1'], obj['x2'], obj['y2']]
            tooth = obj['tooth']
            diagnosis = obj['diagnosis']
            
            treatment_groups[treatment].append({
                'image_path': image_path,
                'image_name': image_name,
                'bbox': bbox,
                'tooth': tooth,
                'treatment': treatment,
                'diagnosis': diagnosis
            })
    
    # Print original distribution
    print("\nOriginal treatment distribution:")
    for t, items in sorted(treatment_groups.items()):
        print(f"  {t}: {len(items)}")
    
    # Balance: sample ~1072 from 'none' to match total of other treatments
    other_count = sum(len(v) for k, v in treatment_groups.items() if k != 'none')
    target_none = min(1072, len(treatment_groups['none']))
    
    print(f"\nOther treatments total: {other_count}")
    print(f"Target 'none' samples: {target_none}")
    
    # Sample from none
    none_samples = random.sample(treatment_groups['none'], target_none)
    treatment_groups['none'] = none_samples
    
    # Flatten all samples
    all_samples = []
    for treatment, samples in treatment_groups.items():
        all_samples.extend(samples)
    
    random.shuffle(all_samples)
    
    print(f"\nTotal samples after balancing: {len(all_samples)}")
    
    # Process and save
    output_dir = 'dataset_treatment_classified'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing {len(all_samples)} samples...")
    
    output_records = []
    
    for idx, sample in enumerate(all_samples):
        # Create unique filename
        image_stem = Path(sample['image_name']).stem
        filename = f"{image_stem}_tooth_{idx:05d}.jpg"
        output_path = os.path.join(output_dir, filename)
        
        # Crop and resize
        crop_and_resize(sample['image_path'], sample['bbox'], output_path)
        
        # Create record
        record = {
            'image': filename,
            'tooth': sample['tooth'],
            'treatment': sample['treatment'],
            'diagnosis': sample['diagnosis']
        }
        output_records.append(record)
        
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{len(all_samples)}...")
    
    # Save jsonl
    output_jsonl = os.path.join(output_dir, 'dataset.jsonl')
    with open(output_jsonl, 'w') as f:
        for record in output_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"\nDone!")
    print(f"Saved {len(output_records)} images to {output_dir}/")
    print(f"Saved metadata to {output_jsonl}")
    
    # Final distribution
    final_dist = defaultdict(int)
    for r in output_records:
        final_dist[r['treatment']] += 1
    
    print("\nFinal treatment distribution:")
    for t, count in sorted(final_dist.items()):
        print(f"  {t}: {count}")

if __name__ == '__main__':
    main()
