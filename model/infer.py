import os
import cv2
import torch
import numpy as np
from unet import UNet


def run_inference(
    input_dir="../data/test/input",
    output_dir="../results/images",
    model_path="unet_best.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Running inference on:", device)

    os.makedirs(output_dir, exist_ok=True)

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    images = sorted(os.listdir(input_dir))
    print(f"📂 Found {len(images)} images for inference")

    for idx, name in enumerate(images):
        img_path = os.path.join(input_dir, name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping unreadable file: {name}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256)) / 255.0

        tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(tensor)

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = (output * 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(output_dir, name), output)

        if idx % 20 == 0:
            print(f"✅ Processed {idx+1}/{len(images)} images")

    print("\n🎉 Inference completed successfully!")


if __name__ == "__main__":
    run_inference()