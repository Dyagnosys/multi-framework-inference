# Use the OpenVINO ONNX Runtime image as base
FROM openvino/onnxruntime_ep_ubuntu20:latest

# Set working directory
WORKDIR /app

# (A) Copy your code and assets into the container
#     If your code is in the same folder as this Dockerfile,
#     you can copy everything. Otherwise, adjust accordingly.
COPY . .

# (B) Install Python dependencies.
#     Some may already be installed in the base image;
#     you can remove any duplicates if they're present.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# (C) If you have a requirements.txt, use it; otherwise, install packages directly.
#     This snippet installs the minimal packages needed for your script.
RUN pip3 install --no-cache-dir \
    numpy \
    onnxruntime \
    opencv-python

# (D) Set environment variables for threading
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# (E) Expose any ports if needed (not strictly required for a batch script)
# EXPOSE 8080

# (F) Set the default command to run your script. 
#     If your file is named "main.py" and contains the `if __name__ == "__main__": main()`,
#     this command will start the pipeline.
CMD ["python3", "main.py"]
