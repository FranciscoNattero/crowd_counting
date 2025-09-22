import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from LCDnet import LCDNet

if __name__ == "__main__":

    # --- config (edit paths if needed) ---
    device = "cuda:1"
    pth_path = "runs/lcdnet_rgbt_DroneRGBT/best.pth"
    rgbs = [
        # "datasets/DroneRGBT/Test/RGB/1.jpg",
        # "datasets/DroneRGBT/Test/RGB/2.jpg",
        "src/inputs/image2.png"
    ]

    out_dir = os.path.join(os.path.dirname(pth_path), "inference_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # --- model ---
    model = LCDNet(in_ch=3).to(device)
    # load weights; sets eval()
    model.load_weights("runs/lcdnet_rgbt_DroneRGBT/best.pth")
    # --- run & show ---
    for rgb_path in rgbs:
        # 1) read RGB for display and size
        rgb_img = Image.open(rgb_path).convert("RGB")
        W, H = rgb_img.size

        # 2) predict (count + density)
        # count, density = model.predict_image(
        #     rgb_path=rgb_path,
        #     modality="rgb",
        # )  # density shape: H/2 x W/2

        count, density_halfres = model.predict_image_tta(
            rgb_path=rgb_path,
            ir_path=None,
            modality="rgb",      # or "rgbt" if you have thermal
            scales=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
            do_hflip=True
        )

        # 3) resize density to input size for overlay
        dm = density_halfres
        dm = dm - dm.min()
        mx = dm.max()
        if mx > 0:
            dm = dm / mx
        dm_img = Image.fromarray((dm * 255.0).astype(np.uint8))
        dm_img = dm_img.resize((W, H), resample=Image.BILINEAR)
        dm_vis = np.asarray(dm_img) / 255.0  # back to [0,1] float


        base = os.path.splitext(os.path.basename(rgb_path))[0]
        save_path = os.path.join(out_dir, f"{base}_overlay_{count:.2f}.png")

        # 4) show side-by-side: left RGB, right RGB overlaid with density
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(rgb_img)
        ax1.set_title(f"Input: {os.path.basename(rgb_path)}")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(rgb_img)
        ax2.imshow(dm_vis, alpha=0.5)  # simple overlay
        ax2.set_title(f"Pred count: {count:.2f}")
        ax2.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        # Optional: also save the normalized density alone
        dm_only_path = os.path.join(out_dir, f"{base}_density.png")
        Image.fromarray((dm * 255.0).astype(np.uint8)).save(dm_only_path)

        print(f"Saved: {save_path}")
        print(f"Saved: {dm_only_path}")