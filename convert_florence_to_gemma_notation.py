import re
import json

def convert_florence_to_chest_imagenome_format(image_id, labels_trname):
    """
    Converts Florence-style object detection labels to Chest ImaGenome scene graph object format.
    
    Args:
        image_id: String identifier for the image (used for object_id generation)
        labels_trname: The concatenated string containing object definitions
                       in format: "name:class<loc_x1><loc_y1><loc_x2><loc_y2>..."
    
    Returns:
        A list of object dictionaries in Chest ImaGenome format
    """
    objects = []
    
    # Find all object matches in the string
    # Pattern matches: name:class<loc_x1><loc_y1><loc_x2><loc_y2>
    pattern = r'([a-zA-Z0-9_]+):([a-zA-Z0-9_]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
    
    for idx, match in enumerate(re.finditer(pattern, labels_trname)):
        name = match.group(1)           # e.g., "upper_right_third_molar"
        classification = match.group(2)  # e.g., "normal", "extraction", "RCT"
        x1 = int(match.group(3))
        y1 = int(match.group(4))
        x2 = int(match.group(5))
        y2 = int(match.group(6))
        
        # Ensure coordinates are ordered: top-left (x1,y1) and bottom-right (x2,y2)
        bbox_x1 = min(x1, x2)
        bbox_x2 = max(x1, x2)
        bbox_y1 = min(y1, y2)
        bbox_y2 = max(y1, y2)
        
        # Calculate width and height
        width = bbox_x2 - bbox_x1
        height = bbox_y2 - bbox_y1
        
        # Generate object_id ( Chest ImaGenome style: <image_id>_<name>_<idx> )
        object_id = f"{image_id}_{name}_{idx}"
        
        # Create object dictionary following Chest ImaGenome format
        obj = {
            'object_id': object_id,
            'x1': bbox_x1,
            'y1': bbox_y1,
            'x2': bbox_x2,
            'y2': bbox_y2,
            'width': width,
            'height': height,
            'bbox_name': name,
            'classification': classification  # Added: your classification field
        }
        
        objects.append(obj)
    
    return objects