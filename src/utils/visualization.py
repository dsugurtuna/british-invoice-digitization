import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from src.schemas.detection import DetectionResult

def draw_detections(
    image: np.ndarray, 
    detections: List[DetectionResult], 
    color_map: Dict[str, Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image: Input image as numpy array (BGR)
        detections: List of DetectionResult objects
        color_map: Optional dictionary mapping class names to BGR colors
        
    Returns:
        Image with drawn detections
    """
    img_copy = image.copy()
    
    if color_map is None:
        # Default colors
        color_map = {
            "Invoice Date": (0, 255, 0),      # Green
            "Invoice Number": (255, 0, 0),    # Blue
            "Vendor Name": (0, 0, 255),       # Red
            "Total Amount": (255, 255, 0),    # Cyan
            "VAT Amount": (255, 0, 255),      # Magenta
            "Line Item": (0, 165, 255)        # Orange
        }
        
    for det in detections:
        box = det.box
        label = det.class_name
        conf = det.confidence
        
        # Get color
        color = color_map.get(label, (255, 255, 255))
        
        # Draw rectangle
        x1, y1, x2, y2 = int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f"{label} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img_copy, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    return img_copy
