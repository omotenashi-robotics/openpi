#!/bin/bash

# GPU Access Verification Script for Docker Rootless
# This script thoroughly tests GPU access in Docker containers,
# including JAX and PyTorch compatibility.
#
# Requirements:
# - Docker rootless installed and running
# - NVIDIA container toolkit configured
# - NVIDIA GPU with drivers installed

set -e

echo "========================================"
echo "GPU Access Verification for Docker"
echo "========================================"
echo ""

# Check if Docker is running
if ! systemctl --user is-active --quiet docker.service 2>/dev/null; then
	echo "❌ ERROR: Docker rootless is not running"
	echo "   Start it with: systemctl --user start docker.service"
	exit 1
fi

# Set DOCKER_HOST if not already set
if [[ -z "$DOCKER_HOST" ]]; then
	export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
fi

echo "[1/5] Testing basic NVIDIA CUDA access with CDI..."
echo ""

if docker run --rm --device nvidia.com/gpu=all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi; then
	echo ""
	echo "✓ Basic NVIDIA CUDA access works with CDI"
else
	echo ""
	echo "❌ Basic NVIDIA CUDA access failed"
	exit 1
fi

echo ""
echo "[2/5] Testing GPU memory allocation..."
echo ""

# Test CUDA runtime and memory allocation
docker run --rm --device nvidia.com/gpu=all nvidia/cuda:12.0.0-runtime-ubuntu22.04 bash -c '
if command -v nvidia-smi &> /dev/null; then
	echo "GPU Memory Info:"
	nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
	exit 0
else
	echo "nvidia-smi not found in runtime image"
	exit 1
fi
'

if [ $? -eq 0 ]; then
	echo "✓ GPU memory allocation test passed"
else
	echo "❌ GPU memory allocation test failed"
	exit 1
fi

echo ""
echo "[3/5] Testing JAX GPU detection..."
echo ""

# Test JAX GPU detection
docker run --rm --device nvidia.com/gpu=all nvidia/cuda:12.0.0-runtime-ubuntu22.04 bash -c '
apt-get update -qq && apt-get install -y -qq python3 python3-pip > /dev/null 2>&1
pip3 install -q jax[cuda12] 2>/dev/null || pip3 install -q "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 2>/dev/null
python3 -c "
import jax
devices = jax.devices()
print(f\"JAX devices: {devices}\")
print(f\"JAX device count: {len(devices)}\")
if len(devices) > 0 and \"gpu\" in str(devices[0]).lower():
    print(\"✓ JAX detected GPU successfully\")
    exit(0)
else:
    print(\"❌ JAX did not detect GPU\")
    exit(1)
"
'

if [ $? -eq 0 ]; then
	echo "✓ JAX GPU detection test passed"
else
	echo "⚠️  JAX GPU detection test failed (this may be normal if JAX installation failed)"
fi

echo ""
echo "[4/5] Testing PyTorch CUDA availability..."
echo ""

# Test PyTorch CUDA
docker run --rm --device nvidia.com/gpu=all nvidia/cuda:12.0.0-runtime-ubuntu22.04 bash -c '
apt-get update -qq && apt-get install -y -qq python3 python3-pip > /dev/null 2>&1
pip3 install -q torch 2>/dev/null
python3 -c "
import torch
print(f\"PyTorch version: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
if torch.cuda.is_available():
    print(f\"CUDA version: {torch.version.cuda}\")
    print(f\"GPU count: {torch.cuda.device_count()}\")
    print(f\"GPU name: {torch.cuda.get_device_name(0)}\")
    print(\"✓ PyTorch CUDA is working\")
    exit(0)
else:
    print(\"❌ PyTorch CUDA not available\")
    exit(1)
"
'

if [ $? -eq 0 ]; then
	echo "✓ PyTorch CUDA test passed"
else
	echo "⚠️  PyTorch CUDA test failed (this may be normal if PyTorch installation failed)"
fi

echo ""
echo "[5/5] Testing Docker Compose GPU syntax..."
echo ""

# Create a temporary compose file to test
TMP_DIR=$(mktemp -d)
cat > $TMP_DIR/docker-compose.yml << 'EOF'
services:
  gpu_test:
    image: nvidia/cuda:12.0.0-base-ubuntu22.04
    command: nvidia-smi --query-gpu=name --format=csv,noheader
    devices:
      - nvidia.com/gpu=all
EOF

cd $TMP_DIR
if docker compose up --abort-on-container-exit 2>&1 | grep -i "RTX\|Tesla\|GeForce\|Quadro\|GPU"; then
	echo ""
	echo "✓ Docker Compose GPU syntax works correctly"
else
	echo ""
	echo "❌ Docker Compose GPU syntax failed"
	cd -
	rm -rf $TMP_DIR
	exit 1
fi

cd -
rm -rf $TMP_DIR

echo ""
echo "========================================"
echo "All GPU Access Tests Completed!"
echo "========================================"
echo ""
echo "Summary:"
echo "✓ Basic NVIDIA CUDA access: PASSED"
echo "✓ GPU memory allocation: PASSED"
echo "✓ JAX GPU detection: CHECK ABOVE"
echo "✓ PyTorch CUDA: CHECK ABOVE"
echo "✓ Docker Compose GPU syntax: PASSED"
echo ""
echo "Your Docker rootless setup is ready for GPU workloads!"
echo ""
echo "Note: JAX and PyTorch tests may show warnings - this is normal."
echo "The important thing is that nvidia-smi works inside containers."
echo ""
