import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageEnhance
from lxml import etree 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
import csv


# -----------------------------
# LCDNet model (lightweight, ~0.05M params with 3ch; ~0.052M with 4ch)
# -----------------------------

class LCDNet(nn.Module):
    """
    Conv1: 32 filters, 5x5, pad=2
    MaxPool(2) -> output becomes H/2 x W/2
    Branch A: 32x3x1 -> 32x1x3 -> 64x3x3
    Branch B: 32x1x3 -> 32x3x1 -> 64x3x3
    Concat -> 128 ch
    Head: 1x1 -> 1 ch (density)
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=5, padding=2, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Top branch
        self.t1 = nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0), bias=True)
        self.t2 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1), bias=True)
        self.t3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)

        # Bottom branch
        self.b1 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1), bias=True)
        self.b2 = nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0), bias=True)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)

        # 1x1 head
        self.head = nn.Conv2d(128, 1, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        # Gaussian init std=0.01
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.pool(x)

        ta = F.relu(self.t1(x), inplace=True)
        ta = F.relu(self.t2(ta), inplace=True)
        ta = F.relu(self.t3(ta), inplace=True)

        tb = F.relu(self.b1(x), inplace=True)
        tb = F.relu(self.b2(tb), inplace=True)
        tb = F.relu(self.b3(tb), inplace=True)

        x = torch.cat([ta, tb], dim=1)
        out = self.head(x)  # [B,1,H/2,W/2]
        return out
    
# ===== Utilities: weight loading & inference =====

    def load_weights(self,
                     pth_path: str,
                     device: Optional[str] = None) -> None:
        """
        Load weights into an existing instance. Leaves the architecture as-is.
        If device is given, also moves the module there. Sets eval() for inference.
        """
        dev = device if device is not None else next(self.parameters()).device
        obj = torch.load(pth_path, map_location=dev)
        state = obj.get("model", obj) if isinstance(obj, dict) else obj
        self.load_state_dict(state, strict=False)
        if device is not None:
            self.to(device)
        self.eval()

    @staticmethod
    def _load_rgb(path: str) -> np.ndarray:
        """Open an RGB image file → float32 CHW tensor in [0,1]."""
        img = Image.open(path).convert("RGB")  # Pillow Image.convert docs. :contentReference[oaicite:2]{index=2}
        arr = np.asarray(img, dtype=np.float32) / 255.0
        chw = arr.transpose(2, 0, 1)  # HWC → CHW
        return chw

    @staticmethod
    def _load_ir(path: str) -> np.ndarray:
        """Open a thermal/infrared image file → float32 1×H×W tensor in [0,1]."""
        img = Image.open(path).convert("L")  # single channel. :contentReference[oaicite:3]{index=3}
        arr = np.asarray(img, dtype=np.float32) / 255.0
        chw = arr[None, ...]  # 1×H×W
        return chw

    @torch.no_grad()  # inference: no grads to save memory/compute. :contentReference[oaicite:4]{index=4}
    def predict_tensor(self, x: torch.Tensor) -> Tuple[float, np.ndarray]:
        """
        x: torch tensor on SAME device as the model, shape [C,H,W].
        Returns (count, density_map_np) where density is H/2×W/2 (float32).
        """
        self.eval()
        if x.dim() != 3:
            raise ValueError(f"expect x [C,H,W], got {tuple(x.shape)}")
        yhat = self(x.unsqueeze(0))              # [1,1,h/2,w/2]
        count = float(yhat.sum().item())
        dm = yhat[0, 0].detach().cpu().numpy().astype(np.float32)
        return count, dm

    @torch.no_grad()
    def predict_image(self,
                      rgb_path: Optional[str] = None,
                      ir_path: Optional[str] = None,
                      modality: str = "rgbt") -> Tuple[float, np.ndarray]:
        """
        Run inference on a SINGLE image (RGB, IR, or RGB+T).
        - modality='rgb'  -> needs rgb_path
        - modality='t'    -> needs ir_path
        - modality='rgbt' -> needs both
        Returns (count, density_map_np).
        """
        modality = modality.lower()
        if modality not in ("rgb", "t", "rgbt"):
            raise ValueError("modality must be one of: rgb, t, rgbt")

        if modality == "rgb":
            if not rgb_path:
                raise ValueError("rgb_path is required for modality='rgb'")
            rgb = LCDNet._load_rgb(rgb_path)
            x = torch.from_numpy(rgb)

        elif modality == "t":
            if not ir_path:
                raise ValueError("ir_path is required for modality='t'")
            ir = LCDNet._load_ir(ir_path)
            x = torch.from_numpy(ir)

        else:  # rgbt
            if not (rgb_path and ir_path):
                raise ValueError("rgb_path and ir_path are required for modality='rgbt'")
            rgb = LCDNet._load_rgb(rgb_path)    # 3×H×W
            ir  = LCDNet._load_ir(ir_path)      # 1×H×W
            if rgb.shape[1:] != ir.shape[1:]:
                raise ValueError(f"RGB and IR sizes differ: {rgb.shape[1:]} vs {ir.shape[1:]}")
            x = torch.from_numpy(np.concatenate([rgb, ir], axis=0))  # 4×H×W

        device = next(self.parameters()).device
        x = x.to(device)
        return self.predict_tensor(x)

    @torch.no_grad()
    def predict_images(self,
                       rgb_paths: Optional[List[str]] = None,
                       ir_paths: Optional[List[str]] = None,
                       modality: str = "rgbt") -> List[Tuple[str, float]]:
        """
        Batch over a LIST of images (no DataLoader needed).
        - For modality='rgb'  -> pass rgb_paths
        - For modality='t'    -> pass ir_paths
        - For modality='rgbt' -> pass both lists, same length & matching order.
        Returns a list of (name, count).
        """
        modality = modality.lower()
        out: List[Tuple[str, float]] = []
        device = next(self.parameters()).device

        if modality == "rgb":
            if not rgb_paths:
                raise ValueError("rgb_paths required for modality='rgb'")
            for p in rgb_paths:
                rgb = LCDNet._load_rgb(p)
                x = torch.from_numpy(rgb).to(device)
                cnt, _ = self.predict_tensor(x)
                out.append((Path(p).name, cnt))
            return out

        if modality == "t":
            if not ir_paths:
                raise ValueError("ir_paths required for modality='t'")
            for p in ir_paths:
                ir = LCDNet._load_ir(p)
                x = torch.from_numpy(ir).to(device)
                cnt, _ = self.predict_tensor(x)
                out.append((Path(p).name, cnt))
            return out

        # rgbt
        if not (rgb_paths and ir_paths):
            raise ValueError("rgb_paths and ir_paths required for modality='rgbt'")
        if len(rgb_paths) != len(ir_paths):
            raise ValueError("rgb_paths and ir_paths must have same length")
        for pr, pt in zip(rgb_paths, ir_paths):
            rgb = LCDNet._load_rgb(pr)
            ir  = LCDNet._load_ir(pt)
            if rgb.shape[1:] != ir.shape[1:]:
                raise ValueError(f"RGB and IR sizes differ: {rgb.shape[1:]} vs {ir.shape[1:]}")
            x = torch.from_numpy(np.concatenate([rgb, ir], axis=0)).to(device)
            cnt, _ = self.predict_tensor(x)
            out.append((Path(pr).name, cnt))
        return out
    
    @torch.no_grad()
    def predict_image_tta(self, rgb_path: str, ir_path: str = None, modality: str = "rgb", scales=(0.75, 1.0, 1.25, 1.5), do_hflip=True):
        """
        Test-time augmentation: multi-scale + optional H-flip.
        Returns (count, density) at the ORIGINAL image size.
        """
        # load base images
        base_rgb = Image.open(rgb_path).convert("RGB")
        W0, H0 = base_rgb.size
        base_ir = Image.open(ir_path).convert("L") if (modality=="t" or modality=="rgbt") and ir_path is not None else None

        device = next(self.parameters()).device
        acc = np.zeros((H0//2, W0//2), dtype=np.float32)  # LCDNet outputs half-res

        for s in scales:
            rgb = base_rgb.resize((int(W0*s), int(H0*s)), Image.BILINEAR)
            ir  = base_ir.resize((int(W0*s), int(H0*s)), Image.BILINEAR) if base_ir is not None else None

            def run_once(img_rgb, img_ir):
                # prepare tensor like your predict_image does
                arr_rgb = np.asarray(img_rgb, dtype=np.float32)/255.0
                chw_rgb = arr_rgb.transpose(2,0,1)  # 3×H×W
                if modality == "rgb":
                    x = torch.from_numpy(chw_rgb)
                elif modality == "t":
                    arr_ir = np.asarray(img_ir, dtype=np.float32)/255.0
                    x = torch.from_numpy(arr_ir[None,...])
                else:
                    if img_ir is None:
                        arr_ir = np.zeros((img_rgb.size[1], img_rgb.size[0]), dtype=np.float32)
                    else:
                        arr_ir = np.asarray(img_ir, dtype=np.float32)/255.0
                    x = torch.from_numpy(np.concatenate([chw_rgb, arr_ir[None,...]], axis=0))
                yhat = self(x.to(device).unsqueeze(0))
                dm = yhat[0,0].detach().cpu().numpy()  # at (H*s)/2 × (W*s)/2
                # resize back to original half-res
                dm_img = Image.fromarray(dm)
                dm_resized = dm_img.resize((W0//2, H0//2), Image.BILINEAR)
                return np.asarray(dm_resized, dtype=np.float32)

            # original
            acc += run_once(rgb, ir)

            # optional hflip
            if do_hflip:
                rgb_f = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                ir_f  = ir.transpose(Image.FLIP_LEFT_RIGHT) if ir is not None else None
                dm_f = run_once(rgb_f, ir_f)
                # flip density back
                acc += dm_f[:, ::-1]

        acc /= (len(scales) * (2 if do_hflip else 1))
        count = float(acc.sum())
        return count, acc
    
# -----------------------------
# Density map utilities
# -----------------------------

def gaussian_kernel2d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    radius = int(truncate * sigma + 0.5)
    size = 2 * radius + 1
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return kernel

def points_to_density_map(points_xy: np.ndarray,
                          h: int, w: int,
                          down: int = 2,
                          sigma: float = 5.0) -> np.ndarray:
    assert down in (1, 2, 4, 8)
    Hs, Ws = h // down, w // down
    dm = np.zeros((Hs, Ws), dtype=np.float32)
    if points_xy.size == 0:
        return dm

    k = gaussian_kernel2d(sigma=sigma)
    kr, kc = k.shape[0] // 2, k.shape[1] // 2

    xs = np.clip((points_xy[:, 0] / down).round().astype(int), 0, Ws - 1)
    ys = np.clip((points_xy[:, 1] / down).round().astype(int), 0, Hs - 1)

    for (x, y) in zip(xs, ys):
        r0, r1 = max(0, y - kr), min(Hs, y + kr + 1)
        c0, c1 = max(0, x - kc), min(Ws, x + kc + 1)

        k_r0 = kr - (y - r0)
        k_r1 = kr + (r1 - y)
        k_c0 = kc - (x - c0)
        k_c1 = kc + (c1 - x)

        dm[r0:r1, c0:c1] += k[k_r0:k_r1, k_c0:k_c1]

    return dm


# -----------------------------
# Dataset
# -----------------------------

class DroneRGBTDataset(Dataset):
    """
    DroneRGBT loader with naming:
      RGB:       N.jpg
      Infrared:  NR.jpg
      GT_:       NR.xml   (XML with head points)
    Default dirs/suffixes match DroneRGBT. Override via CLI if needed.

    sigma_mode: fixed | altitude | knn
    """
    def __init__(self,
                 root: str,
                 split: str = "train",
                 modality: str = "rgb",
                 rgb_dir: str = "RGB",
                 t_dir: str = "Infrared",
                 ann_dir: str = "GT_",
                 t_suffix: str = "R",
                 ann_suffix: str = "R",
                 sigma_mode: str = "fixed",
                 sigma: float = 5.0,
                 altitude_csv: Optional[str] = None,
                 sigma_low: float = 7.0,
                 sigma_medium: float = 5.0,
                 sigma_high: float = 3.0,
                 knn_k: int = 3,
                 use_brightness_contrast: bool = True,
                 hflip_p: float = 0.5):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.modality = modality.lower()
        assert self.modality in ("rgb", "t", "rgbt")

        self.rgb_dir = self.root / split / rgb_dir
        self.t_dir = self.root / split / t_dir
        self.ann_dir = self.root / split / ann_dir

        self.t_suffix = t_suffix
        self.ann_suffix = ann_suffix

        self.items = self._collect_items()

        self.sigma_mode = sigma_mode
        assert self.sigma_mode in ("fixed", "altitude", "knn")
        self.sigma = float(sigma)
        self.knn_k = knn_k

        self.altitude_map: Dict[str, str] = {}
        self.sigma_by_alt = {"low": sigma_low, "medium": sigma_medium, "high": sigma_high}
        if altitude_csv is not None and os.path.isfile(altitude_csv):
            self.altitude_map = self._load_altitude_map(altitude_csv)

        self.use_brightness_contrast = use_brightness_contrast
        self.hflip_p = float(hflip_p)

        # difficulty proxy = gt_count
        self.difficulty = []
        for img_path, ann_path in self.items:
            pts = self._parse_points_xml(ann_path)
            self.difficulty.append(len(pts))

    def _collect_items(self) -> List[Tuple[Path, Path]]:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        items = []
        for p in sorted(self.rgb_dir.glob("*")):
            if p.suffix.lower() not in exts:
                continue
            name = p.stem  # e.g., "1"
            # XML is NR.xml
            ann_xml = self.ann_dir / f"{name}{self.ann_suffix}.xml"  # e.g., 1R.xml
            if not ann_xml.exists():
                # Some releases include GT_ prefix on the file inside GT_ folder too
                alt = self.ann_dir / f"GT_{name}{self.ann_suffix}.xml"
                ann_xml = alt if alt.exists() else ann_xml
            if not ann_xml.exists():
                continue
            items.append((p, ann_xml))
        if not items:
            raise RuntimeError(f"No (RGB, XML) pairs found under {self.rgb_dir} and {self.ann_dir}")
        return items

    def _load_altitude_map(self, csv_path: str) -> Dict[str, str]:
        m = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.lower().startswith("image_name"):
                    continue
                name, grp = line.split(",")[:2]
                m[Path(name).stem] = grp.strip().lower()
        return m

    def _read_image(self, path: Path) -> Image.Image:
        return Image.open(str(path)).convert("RGB")

    def _read_thermal(self, path: Path) -> Image.Image:
        return Image.open(str(path)).convert("L")

    def _parse_points_xml(self, xml_path: Path) -> np.ndarray:
        """
        DroneRGBT GT format you showed:
        <annotation>
        <object> ... <point><x>239</x><y>192</y></point> ... </object> (repeated)
        Returns float32 array Nx2 (x,y).
        """
        try:
            root = etree.parse(str(xml_path)).getroot()
        except Exception:
            return np.zeros((0, 2), dtype=np.float32)

        pts = []
        # strict read: <object>/<point>/<x>,<y>
        for obj in root.findall(".//object"):
            p = obj.find("point")
            if p is None:
                continue
            x = p.findtext("x")
            y = p.findtext("y")
            if x is None or y is None:
                continue
            try:
                pts.append((float(x), float(y)))
            except ValueError:
                pass

        if not pts:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32)

    def _knn_sigma(self, pts: np.ndarray) -> float:
        if pts.shape[0] <= 1:
            return max(1.0, self.sigma)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            dists, _ = tree.query(pts, k=min(self.knn_k + 1, pts.shape[0]))
            mean_nn = np.mean(dists[:, 1:], axis=1).mean()
            return max(1.0, 0.3 * float(mean_nn))
        except Exception:
            return float(self.sigma)

    def _choose_sigma(self, img_stem: str, pts: np.ndarray) -> float:
        if self.sigma_mode == "fixed":
            return float(self.sigma)
        elif self.sigma_mode == "altitude":
            grp = self.altitude_map.get(img_stem, "medium")
            return float(self.sigma_by_alt.get(grp, self.sigma))
        else:
            return self._knn_sigma(pts)

    def _augment(self, img_rgb: Image.Image,
                 img_t: Optional[Image.Image]) -> Tuple[Image.Image, Optional[Image.Image]]:
        if np.random.rand() < self.hflip_p:
            img_rgb = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
            if img_t is not None:
                img_t = img_t.transpose(Image.FLIP_LEFT_RIGHT)
        if self.use_brightness_contrast and np.random.rand() < 0.5:
            img_rgb = ImageEnhance.Brightness(img_rgb).enhance(np.random.uniform(0.8, 1.2))
            img_rgb = ImageEnhance.Contrast(img_rgb).enhance(np.random.uniform(0.8, 1.2))
        return img_rgb, img_t

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rgb_path, ann_path = self.items[idx]
        stem = rgb_path.stem  # e.g., "1"

        img_rgb = self._read_image(rgb_path)
        img_t = None

        if self.modality in ("t", "rgbt"):
            # Infrared is NR.jpg by default
            t_path = self.t_dir / f"{stem}{self.t_suffix}{rgb_path.suffix}"
            if t_path.exists():
                img_t = self._read_thermal(t_path)
            else:
                if self.modality == "t":
                    raise FileNotFoundError(f"Missing thermal for {rgb_path}")

        w, h = img_rgb.size
        pts = self._parse_points_xml(ann_path)

        # augment
        img_rgb, img_t = self._augment(img_rgb, img_t)

        # to tensors
        rgb = torch.from_numpy(np.array(img_rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        if img_t is not None:
            t_np = np.array(img_t, dtype=np.float32)[None, ...] / 255.0
        else:
            t_np = None

        if self.modality == "rgb":
            x = rgb
        elif self.modality == "t":
            x = torch.from_numpy(t_np)
        else:
            if t_np is None:
                t_np = np.zeros((1, rgb.shape[1], rgb.shape[2]), dtype=np.float32)
            x = torch.cat([rgb, torch.from_numpy(t_np)], dim=0)

        sigma = self._choose_sigma(stem + self.ann_suffix, pts)  # altitude maps may key on NR
        dm = points_to_density_map(pts, h=h, w=w, down=2, sigma=sigma)
        dm = torch.from_numpy(dm[None, ...])

        target_count = float(pts.shape[0])

        return {
            "image": x,
            "density": dm,
            "count": torch.tensor([target_count], dtype=torch.float32),
            "name": stem
        }


# -----------------------------
# Metrics
# -----------------------------

def mae(pred_counts: np.ndarray, gt_counts: np.ndarray) -> float:
    return float(np.mean(np.abs(pred_counts - gt_counts)))

def game(pred_dm: torch.Tensor, gt_dm: torch.Tensor, m: int = 2) -> float:
    """
    GAME(m): split map into 4^m regions and sum MAE per region.
    pred_dm, gt_dm: [B,1,H,W]
    """
    B, _, H, W = pred_dm.shape
    r, c = 2 ** m, 2 ** m
    h, w = H // r, W // c
    err = 0.0
    for bi in range(B):
        for i in range(r):
            for j in range(c):
                p = pred_dm[bi, 0, i*h:(i+1)*h, j*w:(j+1)*w].sum().item()
                g = gt_dm[bi, 0, i*h:(i+1)*h, j*w:(j+1)*w].sum().item()
                err += abs(p - g)
    return err / B


# -----------------------------
# Training / Eval
# -----------------------------

def make_loader(ds: DroneRGBTDataset, batch_size: int, shuffle: bool,
                use_curriculum: str = "off") -> DataLoader:
    if use_curriculum == "by_count":
        idxs = np.argsort(np.array(ds.difficulty))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(list(idxs))
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True, drop_last=True)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True, drop_last=True)

def train_one_epoch(model, loader, opt, device):
    model.train()
    loss_fn = nn.MSELoss(reduction='mean')
    running = 0.0
    n_samples = 0
    for batch in tqdm(loader, desc="train", dynamic_ncols=True, leave=False, position=1):
        x = batch["image"].to(device)
        y = batch["density"].to(device)

        opt.zero_grad(set_to_none=True)
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        opt.step()

        running += loss.item() * x.size(0)
        n_samples += x.size(0)
    return running / max(1, n_samples)

@torch.no_grad()
def evaluate(model, loader, device, game_m=2):
    model.eval()
    all_pred = []
    all_gt = []
    game_sum = 0.0
    n = 0
    for batch in tqdm(loader, desc="eval", dynamic_ncols=True, leave=False, position=1):
        x = batch["image"].to(device)
        y = batch["density"].to(device)
        yhat = model(x)

        pred_counts = yhat.sum(dim=(1, 2, 3)).cpu().numpy()
        gt_counts = y.sum(dim=(1, 2, 3)).cpu().numpy()

        all_pred.append(pred_counts)
        all_gt.append(gt_counts)

        game_sum += game(yhat, y, m=game_m) * x.size(0)
        n += x.size(0)

    all_pred = np.concatenate(all_pred) if all_pred else np.array([])
    all_gt = np.concatenate(all_gt) if all_gt else np.array([])
    return {
        "MAE": mae(all_pred, all_gt) if all_pred.size else float("nan"),
        f"GAME{game_m}": game_sum / n if n else float("nan")
    }


def main():
    # ====== CONSTANTS (edit here if needed) ======
    DATA_ROOT   = "datasets/DroneRGBT"
    TRAIN_SPLIT = "Train"
    TEST_SPLIT  = "Test"
    MODALITY    = "rgb"          # "rgb", "t", or "rgbt"
    RGB_DIR     = "RGB"
    T_DIR       = "Infrared"
    ANN_DIR     = "GT_"
    T_SUFFIX    = "R"             # RGB stem + "R" -> Infrared / GT file stem
    ANN_SUFFIX  = "R"

    EPOCHS      = 10000
    BATCH_SIZE  = 16
    LR          = 1e-4
    DEVICE      = "cuda:1"        # e.g., "cpu", "cuda", "cuda:0", "cuda:1"
    OUT_DIR     = "runs/lcdnet_rgbt_DroneRGBT"

    # ====== Setup ======
    os.makedirs(OUT_DIR, exist_ok=True)
    in_ch = 4 if MODALITY == "rgbt" else (3 if MODALITY == "rgb" else 1)
    model = LCDNet(in_ch=in_ch).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Train on Train/, evaluate on Test/
    train_ds = DroneRGBTDataset(
        root=DATA_ROOT, split=TRAIN_SPLIT, modality=MODALITY,
        rgb_dir=RGB_DIR, t_dir=T_DIR, ann_dir=ANN_DIR,
        t_suffix=T_SUFFIX, ann_suffix=ANN_SUFFIX,
        sigma_mode="fixed", sigma=5.0
    )
    test_ds = DroneRGBTDataset(
        root=DATA_ROOT, split=TEST_SPLIT, modality=MODALITY,
        rgb_dir=RGB_DIR, t_dir=T_DIR, ann_dir=ANN_DIR,
        t_suffix=T_SUFFIX, ann_suffix=ANN_SUFFIX,
        sigma_mode="fixed", sigma=5.0,
        hflip_p=0.0, use_brightness_contrast=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )

    # ====== Train/Eval loop ======
    best_mae = float("inf")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}", flush=True)
    for epoch in trange(1, EPOCHS + 1, desc="epochs", dynamic_ncols=True, leave=True):
        tr_loss = train_one_epoch(model, train_loader, opt, DEVICE)
        metrics = evaluate(model, val_loader, DEVICE, game_m=2)
        print(f"Epoch {epoch:03d} | train_loss {tr_loss:.6f} | MAE {metrics['MAE']:.3f} | GAME2 {metrics['GAME2']:.3f}", flush=True)

        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "best_mae": best_mae},
                os.path.join(OUT_DIR, "best.pth")
            )

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "last_model.pth"))


if __name__ == "__main__":
    main()
