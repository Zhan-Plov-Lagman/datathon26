#!/usr/bin/env bash
# Use the CUDA PyTorch supplied by the Runpod template. No model downloads here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 - <<'PY'
import torch, torchvision
assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
assert torchvision.__version__.split('+')[0] == '0.23.0', torchvision.__version__
assert torch.cuda.is_available(), 'The template must provide working CUDA'
print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})
PY
python3 -m venv --system-site-packages .venv
# pip respects the installed CUDA build. Do not use an unconstrained uv resolver:
# it can install a second torch/CUDA stack into the virtual environment.
.venv/bin/python -m pip install --no-cache-dir \
    -r requirements-worker.txt -r requirements-integration.txt pytest==9.1.1
# OpenCV-headless supplies cv2. Avoid installing a second, GUI OpenCV package.
.venv/bin/python -m pip install --no-cache-dir --no-deps \
    ultralytics==8.4.182 ultralytics-thop==2.2.2 \
    nvidia-ml-py==13.615.71 ultralytics-platform==0.1.90
.venv/bin/python - <<'PY'
import torch, torchvision, transformers, bitsandbytes, cv2
assert torch.__version__.split('+')[0] == '2.8.0'
assert torchvision.__version__.split('+')[0] == '0.23.0'
assert torch.cuda.is_available()
print({'torch': torch.__version__, 'transformers': transformers.__version__,
       'bitsandbytes': bitsandbytes.__version__, 'cv2': cv2.__version__})
PY
