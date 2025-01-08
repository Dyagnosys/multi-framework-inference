FROM openvino/onnxruntime_ep_ubuntu20

WORKDIR /app
USER root

ENV MKLDNN_VERBOSE=1
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV KMP_BLOCKTIME=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenblas-base \
    libgomp1 \
    intel-mkl

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
CMD ["python3", "main.py"]