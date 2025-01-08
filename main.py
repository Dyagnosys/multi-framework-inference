import torch
import onnxruntime as ort
import numpy as np
import cv2
import logging
import requests
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs('assets/models', exist_ok=True)
os.makedirs('assets/videos', exist_ok=True)

STATIC_MODEL_PATH = os.path.join('assets', 'models', 'FER_static_ResNet50_AffectNet.onnx')
VIDEO_URL = "https://huggingface.co/spaces/vitorcalvi/dyagnosys-free/resolve/main/assets/videos/fitness.mp4"
BATCH_SIZE = 32

class EmotionPredictor:
    def __init__(self, model_path):
        self.session = self.initialize_inference_session(model_path)
        self.lock = threading.Lock()
        
    def initialize_inference_session(self, model_path):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 4
        
        providers = [('CPUExecutionProvider', {
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cpu_memory_arena_cfg': 'kArenaExtendWay'
        })]
        
        return ort.InferenceSession(model_path, sess_options=options, providers=providers)
        
def preprocess_frame(self, frame):
    # Pre-allocate arrays for efficiency
    resized = np.empty((224, 224, 3), dtype=np.float32)
    cv2.resize(frame, (224, 224), dst=resized)
    resized /= 255.0
    
    # Pre-allocated CHW array
    chw = np.empty((3, 224, 224), dtype=np.float32)
    for i in range(3):
        chw[i] = resized[..., i]
        
    # Pre-computed normalization arrays
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    
    # In-place operations
    np.subtract(chw, mean, out=chw)
    np.divide(chw, std, out=chw)
    return chw
        
    def predict(self, frame):
        preprocessed = self.preprocess_frame(frame)
        with self.lock:
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            result = self.session.run([output_name], {input_name: preprocessed[None, ...]})[0]
        return result[0]

def download_video(url, save_path='assets/videos/input.mp4'):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path

def get_emotion_label(prediction):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[np.argmax(prediction)]

def process_video_chunk(frames, predictor):
    results = []
    for frame in frames:
        result = predictor.predict(frame)
        results.append(result)
    return results

def main():
    start_time = time.time()
    logger.info("Starting emotion recognition pipeline...")
    
    video_path = download_video(VIDEO_URL)
    predictor = EmotionPredictor(STATIC_MODEL_PATH)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter('output.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, 
                         (width, height))
    
    frame_count = 0
    batch_frames = []
    batch_originals = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 10 == 0:
                logger.info(f"Processing frame {frame_count}")
            
            batch_frames.append(frame)
            batch_originals.append(frame.copy())
            
            if len(batch_frames) >= BATCH_SIZE:
                chunk_size = len(batch_frames) // 4
                chunks = [batch_frames[i:i + chunk_size] for i in range(0, len(batch_frames), chunk_size)]
                
                futures = [executor.submit(process_video_chunk, chunk, predictor) for chunk in chunks]
                results = []
                for future in futures:
                    results.extend(future.result())
                
                for orig_frame, result in zip(batch_originals, results):
                    emotion = get_emotion_label(result)
                    cv2.putText(orig_frame, 
                              f"Emotion: {emotion}", 
                              (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              1, 
                              (0, 255, 0), 
                              2)
                    out.write(orig_frame)
                
                batch_frames = []
                batch_originals = []
    
    duration = time.time() - start_time
    logger.info(f"Processed {frame_count} frames in {duration:.2f}s ({frame_count/duration:.2f} fps)")
    cap.release()
    out.release()

if __name__ == "__main__":
    main()