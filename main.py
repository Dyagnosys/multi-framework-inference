import os
import time
import logging
import cv2
import numpy as np
import onnxruntime as ort
from queue import Queue, Empty
from threading import Thread, Event

# ------------------------------------------------------------------------
# 1. ENVIRONMENT & LOGGING CONFIGURATION
# ------------------------------------------------------------------------

# (A) Control OpenMP threading for libraries like oneDNN/MKL (if used)
os.environ['OMP_NUM_THREADS'] = '4'

# (B) Optionally set MKL threads (if MKL is your BLAS library)
os.environ['MKL_NUM_THREADS'] = '4'

# (C) Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------
# 2. GLOBAL SETTINGS
# ------------------------------------------------------------------------
BATCH_SIZE = 4
INPUT_RESOLUTION = (224, 224)  # (height, width)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Normalization constants for typical ImageNet-based models
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ------------------------------------------------------------------------
# 3. EMOTION PROCESSOR CLASS (ONNX + MULTI-THREADING)
# ------------------------------------------------------------------------
class EmotionProcessor:
    """
    This class handles:
      - An ONNX Runtime session with CPU ExecutionProvider.
      - A background thread that:
          * Dequeues frames in batches
          * Preprocesses them (resize, normalize, etc.)
          * Runs inference
          * Outputs results in the correct order
    """
    def __init__(self, model_path):
        # (A) Configure ONNX Runtime session options for CPU
        sess_opts = ort.SessionOptions()
        # Increase optimizations
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Set threading parameters
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 2

        # (B) Specify CPUExecutionProvider with MKL-DNN
        # Adjust "mkldnn_threads" to match your environment if needed
        providers = [
            (
                'CPUExecutionProvider',
                {
                    'enable_mkldnn': True,  # or "1" in older versions
                    'mkldnn_threads': 4
                }
            )
        ]

        # (C) Create the ONNX Runtime inference session
        self.session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)

        # (D) Get input/output layer names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # (E) Queues for frames in and results out
        self.input_queue = Queue(maxsize=64)
        self.output_queue = Queue(maxsize=64)

        # (F) Thread control
        self.stop_event = Event()
        self.processing_thread = None

        # (G) Preallocate a batch buffer (NCHW) to avoid repeated memory allocations
        self.batch_data = np.zeros(
            (BATCH_SIZE, 3, INPUT_RESOLUTION[0], INPUT_RESOLUTION[1]),
            dtype=np.float32
        )

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        - frame_bgr: OpenCV frame (H, W, BGR).
        Returns a (3, H, W) float32 array, normalized and ready for inference.
        """
        # 1) Resize
        if frame_bgr.shape[:2] != INPUT_RESOLUTION:
            frame_bgr = cv2.resize(frame_bgr, INPUT_RESOLUTION, interpolation=cv2.INTER_LINEAR)

        # 2) Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 3) Scale to [0, 1] and convert to float32
        frame_rgb = frame_rgb.astype(np.float32) / 255.0

        # 4) Normalize per channel
        frame_rgb[..., 0] = (frame_rgb[..., 0] - MEAN[0]) / STD[0]
        frame_rgb[..., 1] = (frame_rgb[..., 1] - MEAN[1]) / STD[1]
        frame_rgb[..., 2] = (frame_rgb[..., 2] - MEAN[2]) / STD[2]

        # 5) Transpose to CHW format
        frame_chw = np.transpose(frame_rgb, (2, 0, 1))  # shape: (3, H, W)
        return frame_chw

    def run_inference(self, batch_count: int) -> np.ndarray:
        """
        Runs inference on the first 'batch_count' items in self.batch_data.
        Returns the raw output as a (batch_count, num_classes) numpy array.
        """
        if batch_count == 0:
            return np.array([])
        # Run the model
        results = self.session.run(
            [self.output_name],
            {self.input_name: self.batch_data[:batch_count]}
        )
        return results[0]  # shape: (batch_count, num_classes)

    def process_frames(self):
        """
        The background thread loop:
          - Dequeues frames from input_queue
          - Preprocesses them and copies into batch_data
          - Once batch is full (or queue is empty), runs inference
          - Outputs results to output_queue
        """
        batch_indices = []
        frames_in_batch = 0

        while not self.stop_event.is_set():
            try:
                idx, frame = self.input_queue.get(timeout=0.1)
                preprocessed = self.preprocess_frame(frame)
                self.batch_data[frames_in_batch] = preprocessed
                batch_indices.append(idx)
                frames_in_batch += 1

                # If batch is full, run inference
                if frames_in_batch >= BATCH_SIZE:
                    outputs = self.run_inference(frames_in_batch)
                    for i, out in enumerate(outputs):
                        self.output_queue.put((batch_indices[i], out))
                    batch_indices.clear()
                    frames_in_batch = 0

            except Empty:
                # If queue is empty but we have partial batch
                if frames_in_batch > 0:
                    outputs = self.run_inference(frames_in_batch)
                    for i, out in enumerate(outputs):
                        self.output_queue.put((batch_indices[i], out))
                    batch_indices.clear()
                    frames_in_batch = 0

        # Process leftover frames if stop_event is triggered
        if frames_in_batch > 0:
            outputs = self.run_inference(frames_in_batch)
            for i, out in enumerate(outputs):
                self.output_queue.put((batch_indices[i], out))

    def start(self):
        """
        Starts the background processing thread.
        """
        self.processing_thread = Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()

    def stop(self):
        """
        Signals the thread to stop and waits for it to finish.
        """
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5)

# ------------------------------------------------------------------------
# 4. VIDEO PROCESSING FUNCTION
# ------------------------------------------------------------------------
def process_video(input_path: str, output_path: str, processor: EmotionProcessor) -> int:
    """
    Reads a video file with OpenCV, feeds frames to the processor,
    collects results, annotates frames, and writes them to 'output_path'.
    """
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    frame_buffer = {}
    next_frame_idx = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # no more frames

            frame_count += 1
            if frame_count % 10 == 0:
                logger.info(f"Processing frame {frame_count}")

            # Enqueue this frame
            processor.input_queue.put((frame_count, frame))

            # Collect any available output (non-blocking)
            try:
                while True:
                    idx, result = processor.output_queue.get_nowait()
                    frame_buffer[idx] = (frame, result)
            except Empty:
                pass

            # Write frames in correct order
            while (next_frame_idx + 1) in frame_buffer:
                next_frame_idx += 1
                frm, logits = frame_buffer.pop(next_frame_idx)
                # logits shape might be (num_classes,); pick argmax
                top_idx = int(np.argmax(logits))
                # Validate label index
                if 0 <= top_idx < len(EMOTION_LABELS):
                    emotion = EMOTION_LABELS[top_idx]
                else:
                    emotion = "Unknown"
                # Annotate frame
                cv2.putText(
                    frm,
                    f"Emotion: {emotion}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                out.write(frm)

    finally:
        cap.release()
        out.release()

    return frame_count

# ------------------------------------------------------------------------
# 5. MAIN SCRIPT
# ------------------------------------------------------------------------
def main():
    model_path = 'assets/models/FER_static_ResNet50_AffectNet.onnx'  # Path to your ONNX model
    input_video = 'assets/videos/input.mp4'
    output_video = 'output.mp4'

    start_time = time.time()
    logger.info("Starting emotion recognition pipeline with ONNX (CPU)...")

    # Initialize the ONNX processor
    processor = EmotionProcessor(model_path)
    processor.start()

    try:
        frames_processed = process_video(input_video, output_video, processor)
    finally:
        processor.stop()

    duration = time.time() - start_time
    logger.info(
        f"Processed {frames_processed} frames in {duration:.2f}s "
        f"({frames_processed/duration:.2f} FPS on CPU)."
    )

if __name__ == "__main__":
    main()
