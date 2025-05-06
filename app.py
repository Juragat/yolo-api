import os
import gdown
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

# === Constants ===
MODEL_PATH = "yolov8m-seg.pt"
GDRIVE_ID = "1VDfv8NS_6-t9_AG-gTz1NrllOaRiAEE5"

# === Download model if not found ===
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading via gdown...")
    gdown.download(id=GDRIVE_ID, output=MODEL_PATH, quiet=False)
    print("Download complete.")

print(f"Model size: {os.path.getsize(MODEL_PATH) / 1024**2:.2f} MB")

# === Load YOLO model ===
model = YOLO(MODEL_PATH)

# === Create FastAPI app ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "YOLOv8 API is running"}

@app.get("/status")
def status():
    return {"status": "Server is up and running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        print(f"Received file: {file.filename}")
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        print("File read successfully, converting to image...")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((640, 640))

        print("Image loaded successfully")
        results = model.predict(image, stream=False)
        output = []

        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            for cls, conf, box in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
                output.append({
                    "class": model.names[int(cls)],
                    "confidence": float(conf),
                    "bbox": box.tolist()
                })

        print("Prediction complete, returning results")
        return {"results": output}

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
