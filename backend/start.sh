#!/bin/sh

MODEL_PATH="model/skin_cancer_vit.pth"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading model..."
    wget -O $MODEL_PATH "https://drive.google.com/uc?id=YOUR_FILE_ID"
else
    echo "Model already exists."
fi

echo "Starting FastAPI..."
uvicorn main:app --host 0.0.0.0 --port 8000