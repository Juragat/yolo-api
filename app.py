import os
import gdown
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

MODEL_PATH = "yolov8m-seg.pt"
GDRIVE_ID = "1VDfv8NS_6-t9_AG-gTz1NrllOaRiAEE5"

if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading via gdown...")
    gdown.download(id=GDRIVE_ID, output=MODEL_PATH, quiet=False)
    print("Download complete.")
    
print(f"Model size: {os.path.getsize(MODEL_PATH) / 1024**2:.2f} MB")

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
