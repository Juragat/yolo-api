from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load the YOLOv8 model (local file)
model = YOLO("yolov8m-seg.pt")

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
