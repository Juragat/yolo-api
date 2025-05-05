from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io
import os
import requests

MODEL_PATH = "yolov8m-seg.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1VDfv8NS_6-t9_AG-gTz1NrllOaRiAEE5"

# Automatically download the model if it's not already present
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Download complete.")

# Verify model file size and content
if os.path.getsize(MODEL_PATH) < 50000000:  # Check for a minimum size (50MB)
    print("Warning: Model file size seems too small!")
else:
    print(f"Model file ({MODEL_PATH}) size is valid.")

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    results = model(image)
    output = []

    for r in results:
        for cls, conf, box in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
            output.append({
                "class": model.names[int(cls)],
                "confidence": float(conf),
                "bbox": box.tolist()
            })
    
    return {"results": output}
