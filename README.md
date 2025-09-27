# easylsd

Easy, unified access to multiple line segment detectors for the LipMap project and beyond. Run ScaleLSD, HAWPv3, DeepLSD, or OpenCV LSD with a single CLI and a consistent output format.

- Detectors: ScaleLSD, HAWPv3, DeepLSD, OpenCV LSD
- Unified outputs: lines, junctions, scores, width/height
- Visual overlays: save PNG/PDF with colored lines and junctions
- Data export: JSON (wireframe graph) or TXT (x1 y1 x2 y2)
- Auto-checkpoint: downloads pretrained weights into a cache directory


## Install

Prereqs
- Python 3.8+
- PyTorch 2.0+ (install a build appropriate to your CUDA/MPS/CPU from pytorch.org)

From source

```bash
# In the repo root
git clone https://github.com/cherubicXN/easylsd.git
pip install -e easylsd
```


## Quickstart (CLI)

The package installs a console script `easylsd-detect`.

Basic examples

```bash
# 1) Run OpenCV LSD (no checkpoints needed)
# Saves overlay as PNG into ./out
mkdir -p out
easylsd -i path/to/image.jpg --detector lsd --ext png --saveto out

# 2) Run ScaleLSD on a directory of images (auto-download ckpt if missing)
easylsd -i ./images --pattern "*.jpg" --detector scalelsd --ext pdf --saveto out

# 3) Run DeepLSD and export JSON
# If the checkpoint is missing, the CLI attempts to download it.
easylsd -i path/to/image.jpg --detector deeplsd --ext json --saveto out

# 4) Run HAWPv3 on CPU and save TXT lines
# Provide an explicit checkpoint path if you prefer
# Default cache dir is ~/.cache/lipmap/checkpoints
CKPT=~/.cache/lipmap/checkpoints/hawpv3-fdc5487a.pth
easylsd -i path/to/image.jpg --detector hawpv3 --device cpu --ext txt --saveto out --hawpv3-ckpt "$CKPT"
```

Useful flags
- `-i, --img`: input image or directory
- `--pattern`: glob when input is a directory (default: `*color.jpg`)
- `--detector`/`--mode`: `lsd | deeplsd | hawpv3 | scalelsd` (default: `scalelsd`)
- `-t, --threshold`: painter confidence threshold for overlays (default: 0.05)
- `--width --height`: resize for network input; outputs are rescaled back
- `--ext`: `png | pdf | json | txt` (default: `pdf`)
- `--saveto`: output directory (default: `temp_output/{MODE}`)
- `--device`: `cuda | cpu | mps`
- `--disable-show`: disable interactive windows (matplotlib)
- `--draw-junctions-only`: plot endpoints instead of full segments
- `--use_lsd`/`--use_nms`: ScaleLSD inference toggles
- `--ckpt-dir`: cache directory for checkpoints
- `--scalelsd-ckpt --deeplsd-ckpt --hawpv3-ckpt`: custom checkpoint filenames/paths

Notes
- PNG/PDF overlays require `matplotlib`.
- When `--img` is a directory, make sure your `--pattern` matches your filenames.


## Checkpoints

By default, pretrained weights are stored under `~/.cache/lipmap/checkpoints` and auto-downloaded on first use.

Defaults and sources
- ScaleLSD: `scalelsd-vitbase-v2-train-sa1b.pt`
  - https://huggingface.co/cherubicxn/scalelsd/resolve/main/scalelsd-vitbase-v2-train-sa1b.pt
- HAWPv3: `hawpv3-fdc5487a.pth`
  - https://github.com/cherubicXN/hawp-torchhub/releases/download/HAWPv3/hawpv3-fdc5487a.pth
- DeepLSD: `deeplsd_wireframe.tar`
  - https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_wireframe.tar

Offline usage
- Download checkpoints manually and place them in `--ckpt-dir` (or provide absolute paths via `--*-ckpt`).


## Output formats

- Image overlays (`png`, `pdf`)
  - Colored line segments and junction endpoints, thresholded by `-t/--threshold`.
- JSON (wireframe graph)
  - Keys: `vertices`, `vertices-score`, `edges`, `edges-weights`, `height`, `width`
  - `vertices`: Nx2 junction coordinates; `edges`: Mx2 indices into `vertices`
- TXT
  - Plain text, one line per segment: `x1 y1 x2 y2`


## Minimal Python usage

```python
import cv2
import torch
from easylsd.models import ScaleLSD

# Read grayscale image and prepare a tensor
img = cv2.imread("path/to/image.jpg", 0)
H, W = img.shape[:2]
net_h, net_w = 512, 512

image_t = torch.from_numpy(cv2.resize(img, (net_w, net_h))).float() / 255.0
image_t = image_t[None, None].to("cuda" if torch.cuda.is_available() else "cpu")

# Build model and run inference
model = ScaleLSD(gray_scale=True).eval().to(image_t.device)
meta = {
    "width": W, "height": H, "filename": "", "use_lsd": False, "use_nms": True,
}
with torch.no_grad():
    outputs, _ = model(image_t, meta)

pred = outputs[0]
lines = pred["lines_pred"]        # Nx4 (x1,y1,x2,y2) in original image coords
scores = pred["lines_score"]       # N
junctions = pred["juncs_pred"]     # Kx2
```


## Troubleshooting

- “CUDA requested but not available; falling back to CPU”
  - Ensure your PyTorch install matches your CUDA toolkit; or use `--device cpu`.
- Overlays fail to save with an ImportError
  - Install `matplotlib`, or use `--ext json`/`--ext txt`.
- No images found
  - When using directories, verify `--pattern` matches your filenames (e.g., `--pattern "*.png"`).


## Acknowledgments

This project wraps detectors from their respective authors:
- ScaleLSD (Ant Research)
- HAWPv3 (cherubicXN and collaborators)
- DeepLSD (ETH Zürich CVG)
- OpenCV LSD (OpenCV contributors)

Please cite the corresponding papers and repositories when appropriate.

