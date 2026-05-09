import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import __main__
import warnings

warnings.filterwarnings("ignore")

# --- FIX: DUMMY CLASS FOR PICKLE SECURITY ---
class RecorderMeter:
    def __init__(self):
        pass
__main__.RecorderMeter = RecorderMeter

_original_load = torch.load
def _legacy_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _legacy_load
# --------------------------------------------

from network.my_model import Model_5

def run_ablation_study(test_data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CMNet on {device}...")
    
    model = Model_5(num_class=7, device=device)
    model.load_state_dict(torch.load('weights/rafdb.pth', map_location=device)['state_dict'])
    model.to(device)
    model.eval()

    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading Test Dataset from: {test_data_path}")
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # CRITICAL: Perfect 1-to-1 mapping discovered for RAF-DB
    cmnet_map = {
        '0': 0, # Surprise
        '1': 1, # Fear
        '2': 2, # Disgust
        '3': 3, # Happiness
        '4': 4, # Sadness
        '5': 5, # Anger
        '6': 6  # Neutral
    }

    baseline_correct = 0
    ms_tta_correct = 0 # Multi-Scale TTA
    total_images = len(test_dataset)

    print(f"Starting Multi-Scale Evaluation on {total_images} images...\n")

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating MS-TTA", unit="img"):
            inputs = inputs.to(device)
            
            folder_name = test_dataset.classes[labels.item()].lower()
            if folder_name not in cmnet_map:
                continue
            target_idx = cmnet_map[folder_name]

            # --- 1. BASELINE PASS ---
            outputs_base, _ = model(inputs)
            _, pred_base = torch.max(outputs_base, 1)
            if pred_base.item() == target_idx:
                baseline_correct += 1

            # --- 2. SYMMETRY PASS (Flip) ---
            inputs_flipped = torch.flip(inputs, dims=[3])
            outputs_flipped, _ = model(inputs_flipped)

            # --- 3. SCALE PASS (Zoom) ---
            # Interpolate to 256x256 then take the center 224x224 crop
            # This isolates the core facial features (eyes/mouth) and removes background noise
            inputs_zoomed = F.interpolate(inputs, size=(256, 256), mode='bilinear', align_corners=False)
            inputs_zoomed_cropped = inputs_zoomed[:, :, 16:240, 16:240] 
            outputs_zoomed, _ = model(inputs_zoomed_cropped)

            # --- 4. 3-WAY LATE FUSION (MS-TTA) ---
            # We average the raw logits from all three perspective shifts
            fused_logits = (outputs_base + outputs_flipped + outputs_zoomed) / 3.0
            _, pred_tta = torch.max(fused_logits, 1)
            
            if pred_tta.item() == target_idx:
                ms_tta_correct += 1

    # --- Calculate Metrics ---
    base_acc = (baseline_correct / total_images) * 100
    tta_acc = (ms_tta_correct / total_images) * 100
    delta = tta_acc - base_acc

    print("\n" + "=" * 55)
    print("🏆 FINAL ABLATION RESULTS: MULTI-SCALE SA-TTA 🏆")
    print("=" * 55)
    print(f"Total Images Tested:       {total_images}")
    print(f"Baseline CMNet Accuracy:   {base_acc:.2f}%")
    print(f"CMNet + MS-TTA Accuracy:   {tta_acc:.2f}%")
    print(f"Net Accuracy Improvement:  {'+' if delta >= 0 else ''}{delta:.2f}%")
    print("=" * 55)
    print("Note: MS-TTA utilizes Symmetry (Flip) + Scale (Zoom) Late Fusion.")

if __name__ == "__main__":
    # Ensure your RAF-DB 'test' folder is renamed to 'test_data' in the project root
    run_ablation_study('./test_data')