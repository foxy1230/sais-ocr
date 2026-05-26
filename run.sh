#!/bin/bash
set -euo pipefail

echo "Starting YOLO + EfficientNet OCR inference..."

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${NVIDIA_VISIBLE_DEVICES:-}" ] \
  && [ "${NVIDIA_VISIBLE_DEVICES}" != "all" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "void" ]; then
  export CUDA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}"
fi

echo "===== GPU diagnostics ====="
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
python3 -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')" || true
echo "===== End GPU diagnostics ====="

if [ ! -d "/saisdata" ]; then
  echo "Warning: /saisdata not found"
fi
if [ ! -d "/saisresult" ]; then
  echo "Warning: /saisresult not found; creating"
  mkdir -p /saisresult
fi

python3 /app/src/run_inference.py

PREDICTION_FILE="${OUTPUT_FILE:-/saisresult/prediction.json}"
if [ ! -f "${PREDICTION_FILE}" ]; then
  echo "Error: ${PREDICTION_FILE} not found"
  exit 1
fi

echo "Inference complete! Result: ${PREDICTION_FILE}"
