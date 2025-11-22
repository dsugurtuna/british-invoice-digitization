import torch
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InvoiceDigitizer:
    """
    Production-ready class for extracting fields from British invoices using YOLOv5.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', conf_thres: float = 0.4):
        """
        Initialize the InvoiceDigitizer.
        
        Args:
            model_path (str): Path to the .pt model file.
            device (str): 'cpu' or 'cuda'.
            conf_thres (float): Confidence threshold for detections.
        """
        self.device = torch.device(device)
        self.conf_thres = conf_thres
        self.model_path = model_path
        
        logger.info(f"Loading model from {model_path} on {device}...")
        try:
            # Load model using torch.hub to ensure compatibility
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            self.model.to(self.device)
            self.model.conf = conf_thres
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def process_image(self, image_source: Union[str, np.ndarray]) -> Dict:
        """
        Process a single image and return detected fields.
        
        Args:
            image_source (str or np.ndarray): Path to image or numpy array (BGR).
            
        Returns:
            Dict: Structured data containing detected fields and their bounding boxes.
        """
        results = self.model(image_source)
        
        # Parse results
        # results.pandas().xyxy[0] returns a DataFrame with columns: xmin, ymin, xmax, ymax, confidence, class, name
        df = results.pandas().xyxy[0]
        
        output = {
            "meta": {
                "timestamp": str(np.datetime64('now')),
                "image_source": str(image_source) if isinstance(image_source, (str, Path)) else "memory_buffer"
            },
            "detections": []
        }
        
        for _, row in df.iterrows():
            detection = {
                "label": row['name'],
                "confidence": float(row['confidence']),
                "bbox": [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
            }
            output["detections"].append(detection)
            
        logger.info(f"Detected {len(output['detections'])} fields in image.")
        return output

    def batch_process(self, image_folder: str) -> List[Dict]:
        """
        Process all images in a folder.
        """
        path = Path(image_folder)
        results = []
        for img_file in path.glob('*.jpg'):
            try:
                res = self.process_image(str(img_file))
                results.append(res)
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
        return results

if __name__ == "__main__":
    # Example usage
    # Assuming model is exported to models/royal_audit_v1_best.pt
    MODEL_PATH = "../models/royal_audit_v1_best.pt"
    
    # Check if model exists, otherwise use a placeholder or warn
    if not Path(MODEL_PATH).exists():
        logger.warning(f"Model not found at {MODEL_PATH}. Please run the training notebook first.")
    else:
        digitizer = InvoiceDigitizer(model_path=MODEL_PATH)
        # digitizer.process_image("path/to/invoice.jpg")
