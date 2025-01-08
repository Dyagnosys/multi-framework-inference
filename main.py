import os
import time
import logging
import numpy as np
import cv2
import onnxruntime as ort
from queue import Queue, Empty
from threading import Thread, Event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Revert to original resolution
BATCH_SIZE = 4  # Back to original batch size
INPUT_RESOLUTION = (224, 224)  # Original model input size
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionProcessor:
    def __init__(self, model_path):
        # Enhanced Intel-specific ONNX Runtime configuration
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Use Intel MKL execution provider with optimizations
        providers = [
            ('CPUExecutionProvider', {
                'enable_mkldnn': '1',
                'mkldnn_threads': '2'
            })
        ]
        
        # Initialize inference session
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        
        # Prepare buffers with optimized numpy arrays
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Optimized preprocessing buffers
        self.batch_buffer = np.zeros((BATCH_SIZE, 3, *INPUT_RESOLUTION), dtype=np.float32)
        self.resized_buffer = np.zeros((*INPUT_RESOLUTION, 3), dtype=np.float32)
        
        # Normalized preprocessing constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(-1, 1, 1)
        
        # Threading components
        self.input_queue = Queue(maxsize=64)
        self.output_queue = Queue(maxsize=64)
        self.stop_event = Event()

    def preprocess_frame(self, frame, index):
        # Efficient resizing maintaining aspect ratio
        if frame.shape[:2] != INPUT_RESOLUTION:
            frame = cv2.resize(frame, INPUT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        
        # Faster numpy transformations
        self.batch_buffer[index] = (
            np.transpose(frame, (2,0,1)).astype(np.float32) / 255.0 - self.mean
        ) / self.std

    def run_inference(self, frames, count):
        # Batched inference with optimized input
        return self.session.run(
            [self.output_name], 
            {self.input_name: self.batch_buffer[:count]}
        )[0]

    def process_frames(self):
        batch_frames, batch_indices = [], []
        
        while not self.stop_event.is_set():
            try:
                idx, frame = self.input_queue.get(timeout=0.1)
                batch_frames.append(frame)
                batch_indices.append(idx)
                
                if len(batch_frames) >= BATCH_SIZE:
                    for i, f in enumerate(batch_frames):
                        self.preprocess_frame(f, i)
                    results = self.run_inference(batch_frames, len(batch_frames))
                    for idx, result in zip(batch_indices, results):
                        self.output_queue.put((idx, result))
                    batch_frames.clear()
                    batch_indices.clear()
                    
            except Empty:
                if batch_frames:
                    for i, f in enumerate(batch_frames):
                        self.preprocess_frame(f, i)
                    results = self.run_inference(batch_frames, len(batch_frames))
                    for idx, result in zip(batch_indices, results):
                        self.output_queue.put((idx, result))
                    batch_frames.clear()
                    batch_indices.clear()

    def start(self):
        self.processing_thread = Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()

    def stop(self):
        self.stop_event.set()
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=5)

def process_video(input_path, output_path, processor):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, (width, height))
    
    frame_buffer = {}
    next_frame_idx = 0
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if frame_count % 10 == 0:
                logger.info(f"Processing frame {frame_count}")
                
            processor.input_queue.put((frame_count, frame))
            
            try:
                while True:
                    idx, result = processor.output_queue.get_nowait()
                    frame_buffer[idx] = (frame, result)
            except Empty:
                pass
                
            while next_frame_idx + 1 in frame_buffer:
                next_frame_idx += 1
                frame, result = frame_buffer.pop(next_frame_idx)
                emotion = EMOTION_LABELS[np.argmax(result)]
                cv2.putText(frame, f"Emotion: {emotion}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)
    finally:
        cap.release()
        out.release()
    return frame_count

def main():
    model_path = 'assets/models/FER_static_ResNet50_AffectNet.onnx'
    input_video = 'assets/videos/input.mp4'
    output_video = 'output.mp4'
    
    start = time.time()
    logger.info("Starting emotion recognition pipeline...")
    
    processor = EmotionProcessor(model_path)
    processor.start()
    
    try:
        frames = process_video(input_video, output_video, processor)
    finally:
        processor.stop()
        
    duration = time.time() - start
    logger.info(f"Processed {frames} frames in {duration:.2f}s ({frames/duration:.2f} fps)")

if __name__ == "__main__":
    main()