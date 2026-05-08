from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import warnings
import __main__

warnings.filterwarnings("ignore", category=FutureWarning)

# --- FIX 1: DUMMY CLASS FOR PICKLE SECURITY ---
# The authors saved a custom 'RecorderMeter' class in their weights.
# We must define it here so standard pickle doesn't crash when loading the backbone.
class RecorderMeter:
    def __init__(self):
        pass

# Inject it directly into the __main__ namespace so pickle finds it
__main__.RecorderMeter = RecorderMeter

# --- FIX 2: PYTORCH 2.6 COMPATIBILITY PATCH ---
# The authors hardcoded torch.load inside Model_5 without weights_only=False.
# We globally patch torch.load to always use weights_only=False for this script.
_original_load = torch.load
def _legacy_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _legacy_load
# ----------------------------------------------

# Import the full SOTA architecture from their network folder
# We must import this AFTER the patches above
from network.my_model import Model_5

app = FastAPI(title="CMNet Facial Expression API")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard RAF-DB 7-class configuration
emotion_classes = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

print("Initializing CMNet (Model_5) Architecture...")
model = Model_5(num_class=7, device=device)

print("Loading RAF-DB Fine-tuned Weights...")
weights_path = 'weights/rafdb.pth'
checkpoint = torch.load(weights_path, map_location=device)

# Handle different PyTorch save formats
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# Standard image processing pipeline for 224x224 networks
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 1. Original Image Tensor
        tensor_orig = transform(image).unsqueeze(0).to(device)
        
        # --- OUR NOVEL CONTRIBUTION: SA-TTA ---
        # 2. Generate a horizontally flipped (mirrored) version of the face
        image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        tensor_flipped = transform(image_flipped).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 3. Run BOTH through the network
            outputs_orig, _ = model(tensor_orig)
            outputs_flipped, _ = model(tensor_flipped)
            
            # 4. Mathematically fuse the logits (Late Fusion) to enforce symmetry validation
            fused_logits = (outputs_orig[0] + outputs_flipped[0]) / 2.0
            
            # Calculate final probabilities from the fused logits
            probabilities = torch.nn.functional.softmax(fused_logits, dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            emotion = emotion_classes[predicted_idx.item()]
        # --------------------------------------

        return JSONResponse(content={
            "status": "success",
            "prediction": emotion,
            "confidence": round(float(confidence), 4),
            "architecture": "CMNet w/ Custom SA-TTA Fusion",
            "contribution": "Symmetry-Aware Test-Time Augmentation Applied"
        })
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)