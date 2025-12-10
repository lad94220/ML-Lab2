from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import base64
from core import process
from typing import Optional

app = FastAPI()

origins = [
  "http://localhost:3000",
  "http://localhost:5173",
  "https://handwriting-recognize.vercel.app"
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

def process_image(image: Image.Image):
    """Preprocess MNIST chuẩn: grayscale → crop → 20x20 → pad 28x28"""

    # 1. Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')

    # Convert to numpy
    img = np.array(image).astype(np.uint8)

    # 2. Binarize (simple threshold, MNIST style)
    binary = (img < 128).astype(np.uint8) * 255

    # 3. Auto-invert nếu nền sáng hơn chữ
    white = np.count_nonzero(binary)
    black = binary.size - white
    if white > black:
        binary = 255 - binary

    # 4. Crop vùng có chữ
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        final = np.zeros((28, 28), dtype=np.uint8)
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        digit = binary[y_min:y_max+1, x_min:x_max+1]

        # 5. Resize giữ tỉ lệ về 20×20
        h, w = digit.shape
        scale = 20 / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        digit_img = Image.fromarray(digit)
        digit_small = digit_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Tạo canvas 28×28 và padding
        canvas = Image.new("L", (28, 28), 0)
        x_pad = (28 - new_w) // 2
        y_pad = (28 - new_h) // 2
        canvas.paste(digit_small, (x_pad, y_pad))

        final = np.array(canvas).astype(np.uint8)

    # 6. Add batch dimension (1,28,28)
    final = np.expand_dims(final, axis=0)

    # 7. Gọi model
    return process(final)

@app.get("/")
async def root():
  return {"message": "MNIST Digit Recognition API"}

@app.post("/predict")
async def predict(request: Request, file: Optional[UploadFile] = File(None)):
  try:
    # Check if it's a JSON request (base64)
    content_type = request.headers.get("content-type", "")
    
    if "application/json" in content_type:
      # Handle base64 image
      body = await request.json()
      image_data = body.get("image", "")
      
      # Remove data URL prefix if present
      if ',' in image_data:
        image_data = image_data.split(',', 1)[1]
      
      # Decode base64 to bytes
      image_bytes = base64.b64decode(image_data)
      image = Image.open(io.BytesIO(image_bytes))
      
    elif file:
      # Handle file upload
      if not file.content_type.startswith("image/"):
        return {"error": "File is not an image."}
      
      contents = await file.read()
      image = Image.open(io.BytesIO(contents))
      
    else:
      return {"error": "No image data provided"}
    
    # Process image and get predictions
    results = process_image(image)
    return results
    
  except Exception as e:
    return {"error": str(e)}

@app.get("/models")
async def get_models():
  """Get list of available models"""
  return {
    "models": ["raw", "edges", "pca"]
  }
  