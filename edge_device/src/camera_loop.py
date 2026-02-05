"""
Camera Loop - Raspberry Pi Edge Device
5-second YOLO inference loop with MQTT publishing
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
import argparse

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import camera libraries
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.warning("OpenCV not available - camera capture disabled")

# Try to import MQTT
try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False
    logger.warning("paho-mqtt not available")

# Try to import ONNX runtime for YOLO
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logger.warning("ONNX Runtime not available")


# YOLO class mapping for traffic detection
CLASS_NAMES = {
    0: 'car',
    1: 'bike',
    2: 'hmv',      # Heavy Motor Vehicle
    3: 'auto',     # Auto-rickshaw
    4: 'ambulance'
}


class YOLODetector:
    """YOLO detector using ONNX runtime for Raspberry Pi."""
    
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to ONNX model file
            input_size: Model input size (width, height)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_names: list = []
        
    def load_model(self) -> bool:
        """Load ONNX model."""
        if not HAS_ONNX:
            logger.error("ONNX Runtime not available")
            return False
            
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return False
            
        try:
            # Use CPU provider for Raspberry Pi
            providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers
            )
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            logger.info(f"Loaded YOLO model: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
            
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO inference."""
        # Resize
        resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
        
    def postprocess(self, outputs: np.ndarray, orig_shape: Tuple[int, int]) -> Dict[str, int]:
        """
        Postprocess YOLO outputs.
        
        Returns:
            Dictionary of class_name -> count
        """
        counts = {name: 0 for name in CLASS_NAMES.values()}
        
        # YOLO output format: [batch, num_detections, 5+num_classes]
        # or [batch, 5+num_classes, num_detections]
        predictions = outputs[0] if len(outputs) == 1 else outputs
        
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dim
            
        # Handle transposed output
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
            
        for pred in predictions:
            # Format: [x, y, w, h, conf, class_probs...]
            if len(pred) >= 6:
                confidence = pred[4]
                
                if confidence >= self.conf_threshold:
                    class_probs = pred[5:]
                    class_id = np.argmax(class_probs)
                    
                    if class_id in CLASS_NAMES:
                        counts[CLASS_NAMES[class_id]] += 1
                        
        return counts
        
    def detect(self, image: np.ndarray) -> Dict[str, int]:
        """
        Run detection on image.
        
        Args:
            image: BGR image from camera
            
        Returns:
            Dictionary of class_name -> count
        """
        if self.session is None:
            logger.warning("Model not loaded")
            return {name: 0 for name in CLASS_NAMES.values()}
            
        try:
            # Preprocess
            input_data = self.preprocess(image)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            
            # Postprocess
            counts = self.postprocess(outputs[0], image.shape[:2])
            
            return counts
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {name: 0 for name in CLASS_NAMES.values()}


class CameraCapture:
    """Camera capture for Raspberry Pi."""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera.
        
        Args:
            camera_id: Camera device ID
            width: Capture width
            height: Capture height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        
    def open(self) -> bool:
        """Open camera device."""
        if not HAS_OPENCV:
            logger.error("OpenCV not available")
            return False
            
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
                
            logger.info(f"Camera {self.camera_id} opened: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            logger.error(f"Camera error: {e}")
            return False
            
    def capture(self) -> Optional[np.ndarray]:
        """Capture a frame."""
        if self.cap is None or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning("Failed to capture frame")
            return None
            
        return frame
        
    def close(self):
        """Close camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class EdgeDevice:
    """
    Main edge device class for Raspberry Pi.
    Captures camera frames, runs YOLO inference, publishes to MQTT.
    """
    
    def __init__(
        self,
        camera_id: str,
        intersection_id: str,
        model_path: str,
        mqtt_broker: str = 'localhost',
        mqtt_port: int = 1883,
        capture_interval: float = 5.0
    ):
        """
        Initialize edge device.
        
        Args:
            camera_id: Unique camera identifier (e.g., "cam_0")
            intersection_id: Parent intersection ID
            model_path: Path to YOLO ONNX model
            mqtt_broker: MQTT broker address
            mqtt_port: MQTT broker port
            capture_interval: Seconds between captures
        """
        self.camera_id = camera_id
        self.intersection_id = intersection_id
        self.model_path = model_path
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.capture_interval = capture_interval
        
        # Components
        self.camera = CameraCapture()
        self.detector = YOLODetector(model_path)
        self.mqtt_client: Optional[mqtt.Client] = None
        
        # State
        self.running = False
        self.frame_count = 0
        
    def initialize(self) -> bool:
        """Initialize all components."""
        success = True
        
        # Load YOLO model
        if not self.detector.load_model():
            logger.warning("YOLO model not loaded - using dummy detections")
            
        # Open camera
        if not self.camera.open():
            logger.warning("Camera not available - using dummy frames")
            
        # Connect MQTT
        if HAS_MQTT:
            try:
                self.mqtt_client = mqtt.Client(
                    client_id=f"edge_{self.intersection_id}_{self.camera_id}"
                )
                self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
                self.mqtt_client.loop_start()
                logger.info(f"MQTT connected to {self.mqtt_broker}:{self.mqtt_port}")
            except Exception as e:
                logger.warning(f"MQTT connection failed: {e}")
                self.mqtt_client = None
        else:
            logger.warning("MQTT not available")
            
        return success
        
    def _generate_dummy_detection(self) -> Dict[str, int]:
        """Generate dummy detection for testing."""
        import random
        return {
            'car': random.randint(0, 10),
            'bike': random.randint(0, 5),
            'hmv': random.randint(0, 2),
            'auto': random.randint(0, 8),
            'ambulance': 1 if random.random() < 0.01 else 0  # 1% chance
        }
        
    def capture_and_detect(self) -> Dict:
        """Capture frame and run detection."""
        # Try to capture real frame
        frame = self.camera.capture()
        
        if frame is not None:
            # Run YOLO detection
            counts = self.detector.detect(frame)
        else:
            # Use dummy detection
            counts = self._generate_dummy_detection()
            
        # Build payload
        payload = {
            'camera_id': self.camera_id,
            'timestamp': datetime.utcnow().isoformat(),
            'cars': counts.get('car', 0),
            'bikes': counts.get('bike', 0),
            'hmv': counts.get('hmv', 0),
            'auto': counts.get('auto', 0),
            'ambulance': counts.get('ambulance', 0),
            'ambulance_detected': counts.get('ambulance', 0) > 0
        }
        
        return payload
        
    def publish_snapshot(self, payload: Dict):
        """Publish snapshot to MQTT."""
        topic = f"traffic/edge/{self.intersection_id}/{self.camera_id}/snapshot"
        
        if self.mqtt_client is not None:
            try:
                self.mqtt_client.publish(topic, json.dumps(payload))
                logger.debug(f"Published to {topic}")
            except Exception as e:
                logger.error(f"MQTT publish failed: {e}")
        else:
            # Log to console if no MQTT
            logger.info(f"[NO MQTT] {topic}: {json.dumps(payload)}")
            
    def run(self):
        """Main capture loop."""
        self.running = True
        logger.info(f"Starting camera loop: {self.capture_interval}s interval")
        
        try:
            while self.running:
                start_time = time.time()
                
                # Capture and detect
                payload = self.capture_and_detect()
                
                # Publish
                self.publish_snapshot(payload)
                
                self.frame_count += 1
                
                # Log periodically
                if self.frame_count % 10 == 0:
                    logger.info(
                        f"Frame {self.frame_count}: "
                        f"cars={payload['cars']}, bikes={payload['bikes']}, "
                        f"hmv={payload['hmv']}, auto={payload['auto']}, "
                        f"ambulance={payload['ambulance']}"
                    )
                    
                # Wait for next interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.capture_interval - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        self.camera.close()
        
        if self.mqtt_client is not None:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            
        logger.info("Cleanup complete")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Raspberry Pi Edge Device - YOLO Traffic Detection'
    )
    parser.add_argument(
        '--camera_id',
        type=str,
        default='cam_0',
        help='Unique camera identifier'
    )
    parser.add_argument(
        '--intersection_id',
        type=str,
        required=True,
        help='Parent intersection ID'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='/models/yolo11n_best.onnx',
        help='Path to YOLO ONNX model'
    )
    parser.add_argument(
        '--mqtt_broker',
        type=str,
        default=os.getenv('MQTT_BROKER', 'localhost'),
        help='MQTT broker address'
    )
    parser.add_argument(
        '--mqtt_port',
        type=int,
        default=int(os.getenv('MQTT_PORT', 1883)),
        help='MQTT broker port'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=5.0,
        help='Capture interval in seconds'
    )
    
    args = parser.parse_args()
    
    # Create and run edge device
    device = EdgeDevice(
        camera_id=args.camera_id,
        intersection_id=args.intersection_id,
        model_path=args.model,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        capture_interval=args.interval
    )
    
    if device.initialize():
        device.run()
    else:
        logger.error("Failed to initialize edge device")
        sys.exit(1)


if __name__ == '__main__':
    main()
