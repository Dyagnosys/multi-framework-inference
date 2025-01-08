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

# ---------------------------------------------
# Configuration
# ---------------------------------------------
BATCH_SIZE = 4
INPUT_RESOLUTION = (224, 224)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionProcessor:
    """
    Handles batched emotion recognition using an ONNX model. 
    Manages a queue of incoming frames, preprocesses them, runs inference, 
    and outputs a queue of (frame_index, inference_result) tuples.
    """
    def __init__(self, model_path):
        # (1) Enhanced ONNX Runtime configuration
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # (2) Intel MKL execution provider with optimizations
        providers = [
            ('CPUExecutionProvider', {
                'enable_mkldnn': '1',
                'mkldnn_threads': '2'
            })
        ]

        # (3) Initialize the inference session
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)

        # (4) Extract input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # (5) Preallocate batch buffer for efficient reuse
        #     This avoids re-allocation of arrays during every inference call.
        self.batch_buffer = np.zeros((BATCH_SIZE, 3, *INPUT_RESOLUTION), dtype=np.float32)

        # (6) Normalized preprocessing constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(-1, 1, 1)

        # (7) Threading and synchronization
        self.input_queue = Queue(maxsize=64)
        self.output_queue = Queue(maxsize=64)
        self.stop_event = Event()
        self.processing_thread = None

    def preprocess_frame(self, frame):
        """
        Prepares a single frame for inference:
        1. Resizes to the model’s input resolution (if needed).
        2. Converts to float32 and normalizes (mean/std).
        3. Returns a (3, H, W) numpy array suitable for the model input.
        """
        # Resize if necessary
        if frame.shape[:2] != INPUT_RESOLUTION:
            frame = cv2.resize(frame, INPUT_RESOLUTION, interpolation=cv2.INTER_LINEAR)

        # Convert to float32, normalize, and transpose (H,W,C) -> (C,H,W)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - self.mean.transpose((1, 2, 0))) / self.std.transpose((1, 2, 0)) 
        frame = np.transpose(frame, (2, 0, 1))  # (3, H, W)
        return frame

    def run_inference(self, count):
        """
        Runs inference on the first 'count' elements of the batch buffer.
        Returns a (count x num_classes) array of predictions.
        """
        # Only run inference if we have frames to process
        if count == 0:
            return []
        return self.session.run(
            [self.output_name],
            {self.input_name: self.batch_buffer[:count]}
        )[0]

    def process_frames(self):
        """
        Continuously processes frames from the input queue in batches.
        When the batch size is reached or the queue times out, 
        inference is run on the accumulated frames.
        """
        batch_frames_indices = []
        frames_accumulated = 0

        while not self.stop_event.is_set():
            try:
                idx, frame = self.input_queue.get(timeout=0.1)
                # Preprocess frame immediately to avoid repeated memory allocations
                preprocessed = self.preprocess_frame(frame)
                self.batch_buffer[frames_accumulated] = preprocessed
                batch_frames_indices.append(idx)
                frames_accumulated += 1

                # If we reached a full batch, run inference
                if frames_accumulated >= BATCH_SIZE:
                    results = self.run_inference(frames_accumulated)
                    for out_idx, out_result in zip(batch_frames_indices, results):
                        self.output_queue.put((out_idx, out_result))
                    # Reset for the next batch
                    batch_frames_indices.clear()
                    frames_accumulated = 0

            except Empty:
                # If queue is empty but we still have a partial batch, run inference
                if frames_accumulated > 0:
                    results = self.run_inference(frames_accumulated)
                    for out_idx, out_result in zip(batch_frames_indices, results):
                        self.output_queue.put((out_idx, out_result))
                    batch_frames_indices.clear()
                    frames_accumulated = 0

        # If stopped but still have unprocessed frames in the buffer
        if frames_accumulated > 0:
            results = self.run_inference(frames_accumulated)
            for out_idx, out_result in zip(batch_frames_indices, results):
                self.output_queue.put((out_idx, out_result))

    def start(self):
        """
        Starts the frame-processing thread.
        """
        self.processing_thread = Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()

    def stop(self):
        """
        Signals the processing thread to stop and waits for it to finish.
        """
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5)

def process_video(input_path, output_path, processor):
    """
    Reads frames from the given video, pushes them to the processor’s queue,
    receives annotated frames from the output queue, and writes them out.
    """
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output writer for annotated video
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

            # Push the frame to the processor’s queue
            processor.input_queue.put((frame_count, frame))

            # Collect any available output results without blocking
            try:
                while True:
                    idx, result = processor.output_queue.get_nowait()
                    frame_buffer[idx] = (frame, result)
            except Empty:
                pass

            # Write frames in correct order
            while next_frame_idx + 1 in frame_buffer:
                next_frame_idx += 1
                frm, res = frame_buffer.pop(next_frame_idx)
                # Determine the predicted emotion
                emotion_idx = np.argmax(res)
                emotion = EMOTION_LABELS[emotion_idx] if 0 <= emotion_idx < len(EMOTION_LABELS) else 'Unknown'
                # Annotate the frame
                cv2.putText(frm, f"Emotion: {emotion}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frm)

    finally:
        cap.release()
        out.release()
    return frame_count

def main():
    model_path = 'assets/models/FER_static_ResNet50_AffectNet.onnx'
    input_video = 'assets/videos/input.mp4'
    output_video = 'output.mp4'

    start_time = time.time()
    logger.info("Starting emotion recognition pipeline...")

    # Initialize and start the EmotionProcessor
    processor = EmotionProcessor(model_path)
    processor.start()

    try:
        # Process the video
        frames_processed = process_video(input_video, output_video, processor)
    finally:
        # Make sure we stop the background thread
        processor.stop()

    duration = time.time() - start_time
    logger.info(
        f"Processed {frames_processed} frames in {duration:.2f}s "
        f"({frames_processed/duration:.2f} fps)"
    )

if __name__ == "__main__":
    main()
