FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libopencv-core-dev \
    libglib2.0-0 \
    # for SVM
    libsm6 libxrender1 libxext-dev \
    # for OCR
    tesseract-ocr

COPY . /app
RUN cd /app \
    && git clone https://github.com/fgosc/fgoscdata.git --recurse-submodules \
    && pip install --no-cache -r /app/requirements.txt \
    && python3 makeitem.py \
    && python3 makechest.py \
    && python3 makecard.py \
    && mkdir -p /input \
    && mkdir -p /output

ENTRYPOINT python3 /app/fgosccnt.py --folder /input --out_folder /output watch
