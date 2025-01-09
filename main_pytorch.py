import os
import time
import torch
import logging
import numpy as np
import cv2
import torch.nn as nn
import torchvision.transforms as transforms

from model_architectures import ResNet50

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CPU Optimization Constants
torch.set_num_threads(2)
torch.set_flush_denormal(True)

# Determine device (force CPU)
device = torch.device('cpu')
logger.info(f"Using device: {device}")

# Model and Preprocessing Configuration
STATIC_MODEL_PATH = 'assets/models/FER_static_ResNet50_AffectNet.pt'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
INPUT_SIZE = (224, 224)

class EmotionRecognition:
    def __init__(self, model_path):
        # Load model with CPU optimizations
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self, model_path):
        try:
            model = ResNet50(num_classes=7, channels=3).to(device)
            
            # Attempt to load state dict
            state_dict = torch.load(model_path, map_location=device)
            
            # Use strict=False to handle potential minor mismatches
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            return model
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            return model

    def preprocess_frame(self, frame):
        # Convert OpenCV BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_tensor = self.transform(frame_rgb).unsqueeze(0)
        return input_tensor

    def recognize_emotion(self, frame):
        # Preprocess frame
        input_tensor = self.preprocess_frame(frame)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            emotion_idx = torch.argmax(probabilities, dim=1).item()
        
        return EMOTION_LABELS[emotion_idx], probabilities[0][emotion_idx].item()

def process_video(input_path, output_path, emotion_recognizer):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Emotion recognition
        emotion, confidence = emotion_recognizer.recognize_emotion(frame)
        
        # Annotate frame
        text = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(frame)
        
        # Progress logging
        if frame_count % 10 == 0:
            logger.info(f"Processing frame {frame_count}")
    
    cap.release()
    out.release()
    
    duration = time.time() - start_time
    logger.info(f"Processed {frame_count} frames in {duration:.2f}s ({frame_count/duration:.2f} fps)")
    
    return frame_count

def main():
    input_video = 'assets/videos/input.mp4'
    output_video = 'output.mp4'
    
    # Initialize emotion recognizer
    emotion_recognizer = EmotionRecognition(STATIC_MODEL_PATH)
    
    # Process video
    process_video(input_video, output_video, emotion_recognizer)

if __name__ == "__main__":
    main()