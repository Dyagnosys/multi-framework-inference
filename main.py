import torch
import onnxruntime as ort
import numpy as np
import cv2
import logging
import requests
import os
import time
from queue import Queue, Empty
from threading import Thread, Event
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs('assets/models', exist_ok=True)
os.makedirs('assets/videos', exist_ok=True)

STATIC_MODEL_PATH = os.path.join('assets', 'models', 'FER_static_ResNet50_AffectNet.onnx')
VIDEO_URL = "https://huggingface.co/spaces/vitorcalvi/dyagnosys-free/resolve/main/assets/videos/fitness.mp4"
BATCH_SIZE = 4

class EmotionProcessor:
    def __init__(self, model_path):
        self.session = self._init_inference_session(model_path)
        self.input_queue = Queue(maxsize=32)
        self.output_queue = Queue(maxsize=32)
        self.stop_event = Event()
        self.lock = threading.Lock()
        self.processing_thread = None

        # Pre-allocate buffers
        self.batch_buffer = np.zeros((BATCH_SIZE, 3, 224, 224), dtype=np.float32)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(-1, 1, 1)

    def _init_inference_session(self, model_path):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return ort.InferenceSession(model_path, sess_options=opts)

    def preprocess_frame(self, frame, index):
        resized = cv2.resize(frame, (224, 224))
        self.batch_buffer[index] = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0
        self.batch_buffer[index] = (self.batch_buffer[index] - self.mean) / self.std

    def process_frames(self):
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        batch_frames = []
        batch_indices = []
        
        while not self.stop_event.is_set():
            try:
                idx, frame = self.input_queue.get(timeout=0.1)
                batch_frames.append(frame)
                batch_indices.append(idx)
                
                if len(batch_frames) >= BATCH_SIZE:
                    # Process batch
                    for i, frame in enumerate(batch_frames):
                        self.preprocess_frame(frame, i)
                        
                    with self.lock:
                        results = self.session.run(
                            [output_name], 
                            {input_name: self.batch_buffer[:len(batch_frames)]}
                        )[0]
                    
                    for idx, result in zip(batch_indices, results):
                        self.output_queue.put((idx, result))
                        
                    batch_frames = []
                    batch_indices = []
                    
            except Empty:
                if batch_frames:  # Process remaining frames
                    for i, frame in enumerate(batch_frames):
                        self.preprocess_frame(frame, i)
                        
                    with self.lock:
                        results = self.session.run(
                            [output_name], 
                            {input_name: self.batch_buffer[:len(batch_frames)]}
                        )[0]
                    
                    for idx, result in zip(batch_indices, results):
                        self.output_queue.put((idx, result))
                        
                    batch_frames = []
                    batch_indices = []

    def start(self):
        self.processing_thread = Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5)

def download_video(url, save_path='assets/videos/input.mp4'):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path

def get_emotion(prediction):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[np.argmax(prediction)]

def process_video(input_path, output_path, processor):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_buffer = {}
    next_frame_idx = 0
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 10 == 0:
                logger.info(f"Processing frame {frame_count}")
                
            # Queue frame for processing
            processor.input_queue.put((frame_count, frame))
            
            # Get processed results
            try:
                while True:
                    idx, result = processor.output_queue.get_nowait()
                    frame_buffer[idx] = (frame, result)
            except Empty:
                pass
                
            # Write frames in order
            while next_frame_idx + 1 in frame_buffer:
                next_frame_idx += 1
                frame, result = frame_buffer[next_frame_idx]
                emotion = get_emotion(result)
                cv2.putText(frame, f"Emotion: {emotion}",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)
                del frame_buffer[next_frame_idx]
                
    finally:
        cap.release()
        out.release()
        
    return frame_count

def main():
    start_time = time.time()
    logger.info("Starting emotion recognition pipeline...")
    
    video_path = download_video(VIDEO_URL)
    processor = EmotionProcessor(STATIC_MODEL_PATH)
    processor.start()
    
    try:
        frame_count = process_video(video_path, 'output.mp4', processor)
    finally:
        processor.stop()
        
    duration = time.time() - start_time
    logger.info(f"Processed {frame_count} frames in {duration:.2f}s ({frame_count/duration:.2f} fps)")

if __name__ == "__main__":
    main()