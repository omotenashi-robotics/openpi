#!/bin/bash

# NVIDIA Container Toolkit Installation Script for Docker Rootless
# This script installs and configures the NVIDIA Container Toolkit to enable GPU access
# in rootless Docker containers.
#
# Requirements:
# - Docker installed in rootless mode (run install_docker_ubuntu22.sh first)
# - NVIDIA GPU with drivers installed (verify with: nvidia-smi)
# - Ubuntu 22.04
#
# References:
# - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html

set -e  # Exit on any error

echo "============================================================"
echo "NVIDIA Container Toolkit Installation for Docker Rootless"
echo "============================================================"
echo ""

# Step 1: Verify prerequisites
echo "[1/6] Verifying prerequisites..."

# Check if NVIDIA drivers are installed
if ! command -v nvidia-smi &> /dev/null; then
	echo "❌ ERROR: nvidia-smi not found. Please install NVIDIA drivers first."
	echo "   You can verify driver installation with: nvidia-smi"
	exit 1
fi

echo "✓ NVIDIA drivers found:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1

# Check if Docker is installed and running in rootless mode
if ! command -v docker &> /dev/null; then
	echo "❌ ERROR: Docker not found. Please install Docker in rootless mode first."
	echo "   Run: ./install_docker_ubuntu22.sh"
	exit 1
fi

# Check if Docker is running in rootless mode
if ! systemctl --user is-active --quiet docker.service 2>/dev/null; then
	echo "❌ ERROR: Docker rootless is not running."
	echo "   Start it with: systemctl --user start docker.service"
	echo "   Or install it with: ./install_docker_ubuntu22.sh"
	exit 1
fi

echo "✓ Docker rootless is running"

# Verify DOCKER_HOST is set for rootless
if [[ -z "$DOCKER_HOST" ]]; then
	echo "⚠️  WARNING: DOCKER_HOST environment variable not set."
	echo "   Setting it for current session..."
	export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
fi

echo ""

# Step 2: Install NVIDIA Container Toolkit
echo "[2/6] Installing NVIDIA Container Toolkit..."

# Add NVIDIA's GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
	sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add NVIDIA's repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
	sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
	sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

# Enable experimental packages (required for some features)
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

echo "✓ NVIDIA Container Toolkit installed"
echo ""

# Step 3: Configure CDI (Container Device Interface) - Recommended for rootless
echo "[3/6] Configuring CDI (Container Device Interface)..."

# Create CDI directory if it doesn't exist
mkdir -p $HOME/.config/cdi

# Generate CDI specification for NVIDIA devices
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
echo "✓ CDI configuration generated at /etc/cdi/nvidia.yaml"

# Also create user-level CDI config (some tools prefer this)
nvidia-ctk cdi generate --output=$HOME/.config/cdi/nvidia.yaml 2>/dev/null || \
	sudo nvidia-ctk cdi generate --output=$HOME/.config/cdi/nvidia.yaml
sudo chown $(whoami):$(whoami) $HOME/.config/cdi/nvidia.yaml 2>/dev/null || true

echo "✓ User-level CDI configuration created"
echo ""

# Step 4: Configure Docker rootless to use NVIDIA runtime
echo "[4/6] Configuring Docker rootless for NVIDIA runtime..."

# Create Docker config directory if it doesn't exist
mkdir -p $HOME/.config/docker

# Configure NVIDIA runtime for rootless Docker
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json

echo "✓ Docker daemon.json configured for NVIDIA runtime"
echo ""

# Step 5: Restart Docker rootless daemon
echo "[5/6] Restarting Docker rootless daemon..."

systemctl --user restart docker.service

# Wait for Docker to be ready
sleep 3

if systemctl --user is-active --quiet docker.service; then
	echo "✓ Docker rootless daemon restarted successfully"
else
	echo "❌ ERROR: Failed to restart Docker daemon"
	echo "   Check logs with: journalctl --user -u docker.service"
	exit 1
fi

echo ""

# Step 6: Verify GPU access
echo "[6/6] Verifying GPU access in Docker containers..."

# Test GPU access with CUDA container
echo "Running NVIDIA CUDA test container..."
if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
	echo "✓ GPU access verified successfully!"
	echo ""
	echo "GPU details from container:"
	docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
	echo "❌ ERROR: GPU access test failed"
	echo ""
	echo "Troubleshooting steps:"
	echo "1. Check Docker daemon.json:"
	echo "   cat $HOME/.config/docker/daemon.json"
	echo "2. Check CDI configuration:"
	echo "   ls -la /etc/cdi/nvidia.yaml"
	echo "3. Check Docker logs:"
	echo "   journalctl --user -u docker.service -n 50"
	echo "4. Verify NVIDIA drivers:"
	echo "   nvidia-smi"
	echo ""
	echo "You may need to try the CDI-based GPU access syntax:"
	echo "   docker run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi"
	exit 1
fi

echo ""
echo "============================================================"
echo "NVIDIA Container Toolkit Installation Complete!"
echo "============================================================"
echo ""
echo "GPU access is now enabled in Docker rootless containers."
echo ""
echo "Usage examples:"
echo "  1. Using --gpus flag (Docker Compose V2 syntax):"
echo "     docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi"
echo ""
echo "  2. Using CDI device syntax (alternative):"
echo "     docker run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi"
echo ""
echo "  3. In docker-compose.yml:"
echo "     deploy:"
echo "       resources:"
echo "         reservations:"
echo "           devices:"
echo "             - driver: nvidia"
echo "               count: 1"
echo "               capabilities: [gpu]"
echo ""
echo "Test your setup:"
echo "  cd /home/on-premise-3/workspaces/openpi_ws/src/openpi"
echo "  docker compose -f scripts/docker/compose.yml up --build"
echo ""
