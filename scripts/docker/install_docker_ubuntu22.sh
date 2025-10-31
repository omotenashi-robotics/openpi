#!/bin/bash

# Docker Rootless Installation Script for Ubuntu 22.04
# This script installs Docker in rootless mode as required by the openpi documentation.
# Rootless Docker runs the Docker daemon without root privileges, improving security.
#
# Requirements:
# - Ubuntu 22.04
# - No existing rootful Docker installation
# - No snap Docker or Docker Desktop
#
# Reference: https://docs.docker.com/engine/security/rootless/

set -e  # Exit on any error

echo "=============================================="
echo "Docker Rootless Installation for Ubuntu 22.04"
echo "=============================================="
echo ""

# Step 1: Check for and remove conflicting Docker installations
echo "[1/6] Checking for conflicting Docker installations..."

# Check for snap Docker
if snap list 2>/dev/null | grep -q docker; then
	echo "⚠️  WARNING: Snap Docker detected. This is incompatible with nvidia-container-toolkit."
	echo "    Issue: https://github.com/NVIDIA/nvidia-container-toolkit/issues/154"
	read -p "    Remove snap Docker? (y/N): " -n 1 -r
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]; then
		sudo snap remove docker
		echo "✓ Snap Docker removed"
	else
		echo "❌ Cannot proceed with snap Docker installed. Exiting."
		exit 1
	fi
fi

# Check for Docker Desktop
if dpkg -l 2>/dev/null | grep -q docker-desktop; then
	echo "⚠️  WARNING: Docker Desktop detected. This is incompatible with NVIDIA runtime."
	echo "    Issue: https://github.com/NVIDIA/nvidia-container-toolkit/issues/229"
	read -p "    Remove Docker Desktop? (y/N): " -n 1 -r
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]; then
		sudo apt remove -y docker-desktop
		echo "✓ Docker Desktop removed"
	else
		echo "❌ Cannot proceed with Docker Desktop installed. Exiting."
		exit 1
	fi
fi

# Check for rootful Docker
if systemctl is-active --quiet docker 2>/dev/null; then
	echo "⚠️  WARNING: Rootful Docker daemon is running. Cannot run both rootful and rootless simultaneously."
	echo "    You can either:"
	echo "    1. Remove rootful Docker completely (recommended)"
	echo "    2. Stop and disable rootful Docker (keep installed but not running)"
	read -p "    Choose option (1/2/N to exit): " -n 1 -r
	echo
	if [[ $REPLY == "1" ]]; then
		sudo systemctl stop docker.service docker.socket
		sudo systemctl disable docker.service docker.socket
		sudo apt remove -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
		sudo rm -rf /var/lib/docker /etc/docker
		echo "✓ Rootful Docker removed"
	elif [[ $REPLY == "2" ]]; then
		sudo systemctl stop docker.service docker.socket
		sudo systemctl disable docker.service docker.socket
		echo "✓ Rootful Docker stopped and disabled"
	else
		echo "❌ Cannot proceed with rootful Docker running. Exiting."
		exit 1
	fi
fi

echo "✓ No conflicting Docker installations found"
echo ""

# Step 2: Install prerequisites
echo "[2/6] Installing prerequisites for rootless Docker..."
sudo apt-get update
sudo apt-get install -y \
	ca-certificates \
	curl \
	uidmap \
	dbus-user-session \
	slirp4netns \
	fuse-overlayfs

echo "✓ Prerequisites installed"
echo ""

# Step 3: Install Docker CE packages (binaries needed for rootless setup)
echo "[3/6] Installing Docker CE packages..."

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources
echo \
	"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |
	sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update

# Install Docker CE packages (provides binaries for rootless setup)
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Make sure system Docker daemon is NOT running (we want rootless only)
sudo systemctl stop docker.service docker.socket 2>/dev/null || true
sudo systemctl disable docker.service docker.socket 2>/dev/null || true

echo "✓ Docker CE packages installed"
echo ""

# Step 4: Install and configure Docker in rootless mode
echo "[4/6] Setting up Docker in rootless mode..."

# Run the rootless installation script
dockerd-rootless-setuptool.sh install

echo "✓ Docker rootless installed"
echo ""

# Step 5: Configure environment variables
echo "[5/6] Configuring environment variables..."

# Determine which shell RC file to use
if [ -f "$HOME/.bashrc" ]; then
	SHELL_RC="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
	SHELL_RC="$HOME/.zshrc"
else
	SHELL_RC="$HOME/.profile"
fi

# Add Docker rootless environment variables if not already present
if ! grep -q "DOCKER_HOST=unix://" "$SHELL_RC" 2>/dev/null; then
	echo "" >> "$SHELL_RC"
	echo "# Docker rootless configuration" >> "$SHELL_RC"
	echo "export PATH=\$HOME/bin:\$PATH" >> "$SHELL_RC"
	echo "export DOCKER_HOST=unix://\$XDG_RUNTIME_DIR/docker.sock" >> "$SHELL_RC"
	echo "✓ Environment variables added to $SHELL_RC"
else
	echo "✓ Environment variables already configured in $SHELL_RC"
fi

# Export for current session
export PATH=$HOME/bin:$PATH
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock

echo ""

# Step 6: Enable Docker to start on boot
echo "[6/6] Enabling Docker to start on boot..."

# Enable systemd user service
systemctl --user enable docker.service

# Enable linger so Docker starts even when user is not logged in
sudo loginctl enable-linger $(whoami)

echo "✓ Docker configured to start on boot"
echo ""

# Verify installation
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

# Check if Docker daemon is running
if systemctl --user is-active --quiet docker.service; then
	echo "✓ Docker daemon is running"
else
	echo "⚠️  Docker daemon is not running, attempting to start..."
	systemctl --user start docker.service
	sleep 2
	if systemctl --user is-active --quiet docker.service; then
		echo "✓ Docker daemon started successfully"
	else
		echo "❌ Failed to start Docker daemon"
		echo "   Check logs with: journalctl --user -u docker.service"
		exit 1
	fi
fi

# Test Docker with hello-world
echo ""
echo "Testing Docker with hello-world container..."
if docker run --rm hello-world > /dev/null 2>&1; then
	echo "✓ Docker is working correctly"
else
	echo "❌ Docker test failed"
	echo "   Make sure to reload your shell or run:"
	echo "   export DOCKER_HOST=unix://\$XDG_RUNTIME_DIR/docker.sock"
	exit 1
fi

echo ""
echo "=============================================="
echo "Docker Rootless Installation Complete!"
echo "=============================================="
echo ""
echo "Important notes:"
echo "1. Docker is running in rootless mode (no sudo required)"
echo "2. Environment variables have been added to $SHELL_RC"
echo "3. You need to reload your shell or run:"
echo "   source $SHELL_RC"
echo "4. Docker will start automatically on boot"
echo ""
echo "Next steps:"
echo "- Install NVIDIA container toolkit with: ./install_nvidia_container_toolkit.sh"
echo "- Verify GPU access with: docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi"
echo ""
