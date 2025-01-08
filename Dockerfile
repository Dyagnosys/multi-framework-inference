FROM openvino/onnxruntime_ep_ubuntu20

WORKDIR /app

USER root

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenblas-base \
    libgomp1

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "main.py"]