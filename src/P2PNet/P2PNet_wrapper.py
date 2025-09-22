import argparse

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from CrowdCounting_P2PNet.crowd_datasets import build_dataset
from CrowdCounting_P2PNet.engine import *
from CrowdCounting_P2PNet.models import build_model
import os
import warnings

import argparse
from types import SimpleNamespace
from typing import Tuple, List

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

class P2PNetWrapper:
    """
    Config is expected to provide:
      - device_index: int (e.g., 1 for GPU 1) or None for CPU
      - weight_path: str (path to ./weights/SHTechA.pth or your own)
      - backbone: str = "vgg16_bn"
      - row: int = 2
      - line: int = 2
      - score_threshold: float = 0.5
      - resample: one of {Image.Resampling.LANCZOS, Image.BICUBIC, ...} (optional)
    """

    def __init__(self, config):
        self.cfg = config

        # ---- device ----
        if torch.cuda.is_available() and getattr(config, "device_index", None) is not None:
            torch.cuda.set_device(config.device_index)
            self.device = torch.device(f"cuda:{config.device_index}")
        else:
            self.device = torch.device("cpu")

        # ---- args for model builder (no argparse.parse_args()) ----
        self.args = SimpleNamespace(
            backbone=getattr(config, "backbone", "vgg16_bn"),
            row=getattr(config, "row", 2),
            line=getattr(config, "line", 2),
            weight_path=getattr(config, "weight_path", None),
        )

        # ---- build & load ----
        self.model = build_model(self.args).to(self.device).eval()
        ckpt = torch.load(self.args.weight_path, map_location="cpu")
        # repo checkpoints store under 'model'
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state)

        # ---- transforms ----
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # resampler
        self.resample = getattr(Image, "Resampling").LANCZOS

        self.threshold = float(getattr(config, "score_threshold", 0.5))

    @staticmethod
    def _force_mult_of_128(w, h):
        return (w // 128) * 128, (h // 128) * 128

    def _prep_crop(self, crop_bgr: np.ndarray) -> Tuple[Image.Image, float, float]:
        """
        Takes a crop in BGR (typical YOLO/OpenCV), converts to RGB PIL,
        resizes to multiples of 128, returns PIL image + (sx, sy) scale factors
        to map predictions back to the original crop coordinates.
        """
        # BGR->RGB
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb).convert("RGB")

        w0, h0 = pil_img.size
        w1, h1 = self._force_mult_of_128(w0, h0)
        # avoid zero-dimension if tiny
        w1 = max(128, w1)
        h1 = max(128, h1)

        if (w1, h1) != (w0, h0):
            pil_img = pil_img.resize((w1, h1), resample=self.resample)

        sx = w0 / float(w1)
        sy = h0 / float(h1)
        return pil_img, sx, sy

    @torch.inference_mode()
    def infer_on_crop(self, crop_bgr):
        """
        Run P2PNet on a single crop (BGR np array).
        Returns:
          - drawn BGR crop with red dots
          - predicted count
          - list of (x, y) points in *crop coordinates* (float)
        """
        pil_img, sx, sy = self._prep_crop(crop_bgr)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)  # [1,3,H,W]

        outputs = self.model(tensor)
        # logits: [B, Q, 2] -> prob of "person" class at [:, :, 1]
        scores = F.softmax(outputs["pred_logits"], dim=-1)[0, :, 1]
        points = outputs["pred_points"][0]  # [Q, 2], in resized-crop coords

        keep = scores > self.threshold
        points_kept = points[keep].detach().cpu().numpy()

        # map back to the original (pre-resize) crop coordinates
        if len(points_kept) > 0:
            points_kept[:, 0] *= sx
            points_kept[:, 1] *= sy

        # draw
        drawn = crop_bgr.copy()
        for x, y in points_kept:
            cv2.circle(drawn, (int(x), int(y)), 2, (0, 0, 255), -1)

        return drawn, int(keep.sum().item()), [(float(x), float(y)) for x, y in points_kept]

    @torch.inference_mode()
    def infer_on_bbox(self, full_bgr: np.ndarray, xyxy: Tuple[int, int, int, int]) -> Tuple[np.ndarray, int, List[Tuple[float, float]]]:
        """
        Convenience: crop from a full image using xyxy = (x1,y1,x2,y2) in pixels,
        run P2PNet on that crop, and return:
          - drawn *full* image with dots over the bbox region
          - predicted count
          - list of absolute (x, y) points in the full image coordinates
        """
        x1, y1, x2, y2 = map(int, xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        crop = full_bgr[y1:y2, x1:x2].copy()
        drawn_crop, count, pts_crop = self.infer_on_crop(crop)

        # paste crop overlay back (optional visual)
        out = full_bgr.copy()
        out[y1:y2, x1:x2] = drawn_crop

        # map crop points to absolute coords
        pts_abs = [(x1 + x, y1 + y) for (x, y) in pts_crop]
        return out, count, pts_abs