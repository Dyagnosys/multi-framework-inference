import torch, onnxruntime as ort, numpy as np, cv2, logging, os, time
from queue import Queue, Empty, Full
from threading import Thread, Event, Lock

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
        self.batch_buffer = np.zeros((BATCH_SIZE, 3, 224, 224), dtype=np.float32)
        self.resized_buffer = np.zeros((224, 224, 3), dtype=np.float32)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(-1, 1, 1)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _init_inference_session(self, model_path):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return ort.InferenceSession(model_path, sess_options=opts)

    def preprocess_frame(self, frame, index):
        cv2.resize(frame, (224, 224), dst=self.resized_buffer)
        self.batch_buffer[index] = np.transpose(self.resized_buffer, (2,0,1)).astype(np.float32) / 255.0
        self.batch_buffer[index] = (self.batch_buffer[index] - self.mean) / self.std

    def run_inference(self, frames, count):
        for i, frame in enumerate(frames):
            self.preprocess_frame(frame, i)
        return self.session.run([self.output_name], {self.input_name: self.batch_buffer[:count]})[0]

    def process_frames(self):
        batch_frames, batch_indices = [], []
        
        while not self.stop_event.is_set():
            try:
                idx, frame = self.input_queue.get(timeout=0.1)
                batch_frames.append(frame)
                batch_indices.append(idx)
                
                if len(batch_frames) >= BATCH_SIZE:
                    results = self.run_inference(batch_frames, len(batch_frames))
                    for idx, result in zip(batch_indices, results):
                        self.output_queue.put((idx, result))
                    batch_frames.clear()
                    batch_indices.clear()
                    
            except Empty:
                if batch_frames:
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
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, (int(cap.get(3)), int(cap.get(4))))
    
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
                emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][np.argmax(result)]
                cv2.putText(frame, f"Emotion: {emotion}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)
    finally:
        cap.release()
        out.release()
    return frame_count

def main():
    start = time.time()
    logger.info("Starting emotion recognition pipeline...")
    
    processor = EmotionProcessor(STATIC_MODEL_PATH)
    processor.start()
    
    try:
        frames = process_video('assets/videos/input.mp4', 'output.mp4', processor)
    finally:
        processor.stop()
        
    logger.info(f"Processed {frames} frames in {time.time()-start:.2f}s ({frames/(time.time()-start):.2f} fps)")

if __name__ == "__main__":
    main()