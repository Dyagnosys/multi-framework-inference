import torch
import onnxruntime as ort
import numpy as np
import cv2
from model_architectures import ResNet50, LSTMPyTorch
import logging
import requests
import os
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs('assets/models', exist_ok=True)
os.makedirs('assets/videos', exist_ok=True)

STATIC_MODEL_PATH = os.path.join('assets', 'models', 'FER_static_ResNet50_AffectNet.onnx')
DYNAMIC_MODEL_PATH = os.path.join('assets', 'models', 'FER_dynamic_LSTM.onnx')
VIDEO_URL = "https://huggingface.co/spaces/vitorcalvi/dyagnosys-free/resolve/main/assets/videos/fitness.mp4"
BATCH_SIZE = 16

def download_video(url, save_path='assets/videos/input.mp4'):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path

def initialize_inference_session():
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.intra_op_num_threads = 4
    session_options.inter_op_num_threads = 1

    try:
        static_session = ort.InferenceSession(STATIC_MODEL_PATH, 
                                            sess_options=session_options,
                                            providers=['OpenVINOExecutionProvider'])
        return static_session
    except Exception as e:
        logger.error(f"Failed to initialize sessions: {e}")
        return None

class FrameProcessor:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.batch_buffer = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
        
    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (224, 224))
        float_frame = resized.astype(np.float32) / 255.0
        chw = np.transpose(float_frame, (2, 0, 1))
        normalized = (chw - self.mean.reshape(3,1,1)) / self.std.reshape(3,1,1)
        return normalized

    def process_frames(self, frames, session):
        if not frames:
            return []
            
        for i, frame in enumerate(frames):
            self.batch_buffer[i] = self.preprocess_frame(frame)
            
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        results = session.run([output_name], {input_name: self.batch_buffer[:len(frames)]})[0]
        return results

def get_emotion_label(prediction):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[np.argmax(prediction)]

def process_batch(batch_frames, batch_originals, processor, session):
    results = processor.process_frames(batch_frames, session)
    processed_frames = []
    for orig_frame, result in zip(batch_originals, results):
        emotion = get_emotion_label(result)
        cv2.putText(orig_frame, f"Emotion: {emotion}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        processed_frames.append(orig_frame)
    return processed_frames

def main():
    start_time = time.time()
    logger.info("Starting emotion recognition pipeline...")
    
    video_path = download_video(VIDEO_URL)    
    session = initialize_inference_session()
    if not session:
        return
        
    processor = FrameProcessor(BATCH_SIZE)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, (width, height))
    
    frame_count = 0
    batch_frames = []
    batch_originals = []
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 10 == 0:
                logger.info(f"Processing frame {frame_count}")
                
            batch_frames.append(frame)
            batch_originals.append(frame.copy())
            
            if len(batch_frames) == BATCH_SIZE:
                if future:
                    for processed_frame in future.result():
                        out.write(processed_frame)
                        
                future = executor.submit(process_batch, 
                                      batch_frames, 
                                      batch_originals, 
                                      processor, 
                                      session)
                batch_frames = []
                batch_originals = []
        
        if batch_frames and future:
            for processed_frame in future.result():
                out.write(processed_frame)
            
            final_future = executor.submit(process_batch, 
                                        batch_frames, 
                                        batch_originals, 
                                        processor, 
                                        session)
            
            for processed_frame in final_future.result():
                out.write(processed_frame)
    
    duration = time.time() - start_time
    logger.info(f"Processed {frame_count} frames in {duration:.2f}s ({frame_count/duration:.2f} fps)")
    cap.release()
    out.release()

if __name__ == "__main__":
    import time
    main()