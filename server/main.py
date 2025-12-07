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
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

def process_image(image: Image.Image):
  """Common image processing logic - normalize to 28x28 grayscale"""
  # Convert to grayscale if needed
  if image.mode != 'L':
    image = image.convert('L')
  
  # Always resize to ensure 28x28 regardless of input size
  if image.size != (28, 28):
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
  
  # Convert to numpy array
  img_array = np.array(image, dtype=np.uint8)
  
  # Ensure values are in range [0, 255]
  img_array = np.clip(img_array, 0, 255).astype(np.uint8)
  
  # Add batch dimension: (28, 28) -> (1, 28, 28)
  img_array = np.expand_dims(img_array, axis=0)
  
  # Process with models and get predictions
  return process(img_array)

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
  