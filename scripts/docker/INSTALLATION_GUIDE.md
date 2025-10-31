# Docker Rootless Installation Guide for OpenPI

This comprehensive guide provides step-by-step instructions for installing Docker in rootless mode with NVIDIA GPU support for the openpi project, including limitations, workarounds, and troubleshooting.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Detailed Installation Steps](#detailed-installation-steps)
5. [What is Rootless Docker?](#what-is-rootless-docker)
6. [Rootless Mode Limitations and Workarounds](#rootless-mode-limitations-and-workarounds)
7. [Verifying Installation](#verifying-installation)
8. [Troubleshooting](#troubleshooting)
9. [What Changed from Original Scripts](#what-changed-from-original-scripts)
10. [Next Steps](#next-steps)

## Overview

These updated scripts install:
1. **Docker in rootless mode** - Docker daemon runs without root privileges
2. **NVIDIA Container Toolkit** - Enables GPU access inside containers
3. **CDI (Container Device Interface)** - Modern GPU device management

## Prerequisites

- Ubuntu 22.04 LTS
- NVIDIA GPU with drivers installed (verify with `nvidia-smi`)
- No existing Docker installations (snap, Docker Desktop, or rootful Docker)
- Sudo access for initial setup

## Quick Start

```bash
cd /home/on-premise-3/workspaces/openpi_ws/src/openpi/scripts/docker

# Step 1: Install Docker in rootless mode
bash install_docker_ubuntu22.sh

# Step 2: Reload your shell to apply environment variables
source ~/.bashrc  # or source ~/.zshrc if using zsh

# Step 3: Install NVIDIA Container Toolkit
bash install_nvidia_container_toolkit.sh

# Step 4: Verify GPU access
bash verify_gpu_access.sh

# Step 5: Test with openpi
cd ../..
docker compose -f scripts/docker/compose.yml up --build
```

## Detailed Installation Steps

### Step 1: Install Docker Rootless

The `install_docker_ubuntu22.sh` script will:
- ✅ Check for and remove conflicting installations (snap Docker, Docker Desktop)
- ✅ Install prerequisites (uidmap, slirp4netns, fuse-overlayfs, dbus-user-session)
- ✅ Install Docker CE packages
- ✅ Configure Docker in rootless mode
- ✅ Set up environment variables in your shell RC file
- ✅ Enable Docker to start on boot
- ✅ Verify installation with hello-world container

**Run the script**:
```bash
bash scripts/docker/install_docker_ubuntu22.sh
```

**Expected output**:
```
==============================================
Docker Rootless Installation for Ubuntu 22.04
==============================================

[1/6] Checking for conflicting Docker installations...
✓ No conflicting Docker installations found

[2/6] Installing prerequisites for rootless Docker...
✓ Prerequisites installed

[3/6] Installing Docker CE packages...
✓ Docker CE packages installed

[4/6] Setting up Docker in rootless mode...
✓ Docker rootless installed

[5/6] Configuring environment variables...
✓ Environment variables added to ~/.bashrc

[6/6] Enabling Docker to start on boot...
✓ Docker configured to start on boot

==============================================
Verifying installation...
==============================================
✓ Docker daemon is running
✓ Docker is working correctly

==============================================
Docker Rootless Installation Complete!
==============================================
```

**After installation**:
- Reload your shell: `source ~/.bashrc` (or restart your terminal)
- Docker will now run without sudo
- Environment variable `DOCKER_HOST` will be set automatically

### Step 2: Install NVIDIA Container Toolkit

The `install_nvidia_container_toolkit.sh` script will:
- ✅ Verify NVIDIA drivers are installed
- ✅ Verify Docker rootless is running
- ✅ Install nvidia-container-toolkit
- ✅ Configure CDI (Container Device Interface)
- ✅ Configure Docker rootless for NVIDIA runtime
- ✅ Restart Docker daemon
- ✅ Verify GPU access with test container

**Run the script**:
```bash
bash scripts/docker/install_nvidia_container_toolkit.sh
```

**Expected output**:
```
============================================================
NVIDIA Container Toolkit Installation for Docker Rootless
============================================================

[1/6] Verifying prerequisites...
✓ NVIDIA drivers found: NVIDIA GeForce RTX 4080, 580.95.05
✓ Docker rootless is running

[2/6] Installing NVIDIA Container Toolkit...
✓ NVIDIA Container Toolkit installed

[3/6] Configuring CDI (Container Device Interface)...
✓ CDI configuration generated at /etc/cdi/nvidia.yaml
✓ User-level CDI configuration created

[4/6] Configuring Docker rootless for NVIDIA runtime...
✓ Docker daemon.json configured for NVIDIA runtime

[5/6] Restarting Docker rootless daemon...
✓ Docker rootless daemon restarted successfully

[6/6] Verifying GPU access in Docker containers...
✓ GPU access verified successfully!

GPU details from container:
NVIDIA GeForce RTX 4080, 580.95.05, 16376 MiB

============================================================
NVIDIA Container Toolkit Installation Complete!
============================================================
```

### Step 3: Verify GPU Access (Optional but Recommended)

The `verify_gpu_access.sh` script performs comprehensive GPU testing:
- Basic NVIDIA CUDA access (nvidia-smi)
- GPU memory allocation
- JAX GPU detection
- PyTorch CUDA availability
- Docker Compose GPU syntax

**Run the script**:
```bash
bash scripts/docker/verify_gpu_access.sh
```

This will run several GPU tests to ensure everything is working correctly.

### Step 4: Test with OpenPI

Now test the full openpi setup:

```bash
cd /home/on-premise-3/workspaces/openpi_ws/src/openpi

# Test the main policy server
docker compose -f scripts/docker/compose.yml up --build

# Test LIBERO example (if you want GUI rendering)
xhost +local:docker  # Grant X11 access
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

## What Changed from Original Scripts

### Original `install_docker_ubuntu22.sh`
- ❌ Installed rootful Docker (requires sudo for docker commands)
- ❌ Added user to docker group (rootful approach)
- ❌ Enabled system-level Docker service
- ❌ Did not install rootless prerequisites

### Updated `install_docker_ubuntu22.sh`
- ✅ Installs Docker in rootless mode (no sudo needed)
- ✅ Checks for and removes conflicting installations
- ✅ Installs all rootless prerequisites
- ✅ Configures environment variables automatically
- ✅ Enables user systemd service for Docker
- ✅ Enables linger for automatic startup
- ✅ Comprehensive verification

### Original `install_nvidia_container_toolkit.sh`
- ❌ Configured for rootful Docker only
- ❌ Restarted system Docker service
- ❌ No CDI configuration
- ❌ No verification step

### Updated `install_nvidia_container_toolkit.sh`
- ✅ Verifies rootless Docker is running first
- ✅ Configures for rootless Docker (user-level daemon.json)
- ✅ Generates CDI configuration (recommended for rootless)
- ✅ Restarts user Docker service (not system)
- ✅ Verifies GPU access with test container
- ✅ Provides troubleshooting steps if verification fails

## Key Differences: Rootless vs Rootful Docker

| Feature | Rootful Docker | Rootless Docker |
|---------|----------------|-----------------|
| Requires sudo | Yes | No |
| Docker group | Required | Not used |
| Daemon runs as | root | Regular user |
| Systemd service | `systemctl` | `systemctl --user` |
| Config location | `/etc/docker/` | `~/.config/docker/` |
| Socket location | `/var/run/docker.sock` | `$XDG_RUNTIME_DIR/docker.sock` |
| Security | Lower | Higher |
| GPU access | Supported | Supported (with CDI) |

## Environment Variables

After installation, these variables are set in your shell RC file:

```bash
export PATH=$HOME/bin:$PATH
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
```

**Important**: You need to reload your shell after installation:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

Or simply restart your terminal.

## Verifying Installation

### Check Docker is running
```bash
systemctl --user status docker.service
```

### Check Docker version
```bash
docker version
```

### Check GPU access
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Check environment variables
```bash
echo $DOCKER_HOST
echo $PATH | grep "$HOME/bin"
```

## What is Rootless Docker?

Rootless Docker runs the Docker daemon and containers without root privileges. This improves security by:
- Running the daemon as a regular user
- Containers cannot gain root privileges on the host
- Reduced attack surface for container escapes
- Better multi-user systems where each user can run their own Docker daemon
- Compliance with security requirements that prohibit root access

## Rootless Mode Limitations and Workarounds

### Overview of Limitations

Rootless mode has minimal performance overhead for most workloads:
- ✅ GPU performance: **No degradation** (same as rootful)
- ✅ CPU performance: **~1-2% overhead** (negligible)
- ⚠️ Network: **Slightly slower** with slirp4netns (use host mode to avoid)
- ✅ Storage: **Same performance** with fuse-overlayfs

### 1. `privileged: true` - Limited Functionality

**Status**: ⚠️ **Partial Support**

The `privileged: true` flag in Docker Compose grants containers extensive permissions on the host system. In rootless mode:

- ✅ **Works**: Most container operations work fine
- ❌ **Doesn't work**: Cannot grant actual root privileges (by design)
- ⚠️ **May fail**: Some operations that require true root access will fail

**Affected compose files in openpi**:
- `examples/aloha_sim/compose.yml` (line 14)
- `examples/aloha_real/compose.yml` (lines 16, 31, 39)
- `examples/libero/compose.yml` (line 14)

**Why these examples use `privileged: true`**:
- **X11 display access** (`/tmp/.X11-unix` volume mount in libero)
- **EGL/OpenGL rendering** (MuJoCo simulation rendering)
- **Possible device access** (cameras, robot controllers in aloha_real)

**Recommendation**: Try without `privileged: true` first. Most operations should work in rootless mode. Only add it if you encounter specific permission errors.

**Workarounds if needed**:
```yaml
# Remove privileged: true
# Add specific capabilities instead:
cap_add:
  - SYS_ADMIN  # For certain filesystem operations
  - SYS_NICE   # For process priority
```

### 2. GPU Access - ✅ **Fully Supported**

GPU access via `nvidia-container-toolkit` **works perfectly** in rootless mode when properly configured:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

This is handled by CDI (Container Device Interface) and does not require `privileged: true`.

### 3. X11 Display Access - ✅ **Works**

Mounting X11 sockets for GUI rendering works in rootless mode:

```yaml
volumes:
  - /tmp/.X11-unix:/tmp/.X11-unix:ro
environment:
  - DISPLAY=$DISPLAY
```

**Setup required**:
```bash
# Grant access to X11 server
xhost +local:docker

# Or for better security:
xhost +local:$(whoami)
```

This works fine in rootless mode and does not require `privileged: true`.

### 4. Network Mode Host - ✅ **Works with Limitations**

```yaml
network_mode: host
```

**Status**: Works in rootless mode but containers cannot bind to privileged ports (<1024).

All openpi compose files use `network_mode: host`, which should work fine since they likely use ports >= 1024.

### 5. Device Access - ⚠️ **May Require Workarounds**

Direct device access (e.g., cameras, USB devices) in rootless mode:

```yaml
devices:
  - /dev/video0:/dev/video0
```

**Workarounds**:
- Use `--device` flag with proper permissions
- Ensure user has access to device groups (e.g., `video`, `dialout`)
- May need udev rules for some devices
- For `aloha_real`: Ensure user is in appropriate groups (`dialout`, `video`)

### Testing Compose Files with Rootless Mode

**Recommended Approach**:

1. **Try without modifications first**
   ```bash
   docker compose -f examples/libero/compose.yml up --build
   ```

2. **If you encounter permission errors**:
   - Check the error message carefully
   - Determine what specific permission is needed
   - Apply targeted workarounds

3. **Last resort**: Use `--security-opt` flags (use carefully!)
   ```yaml
   security_opt:
     - "no-new-privileges:true"
     - "seccomp=unconfined"  # Use with caution
   ```

### Specific Workarounds for OpenPI Examples

#### For `libero` example:
```bash
# Grant X11 access
xhost +local:docker

# Run normally - privileged:true should work or can be removed
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

#### For `aloha_sim` example:
```bash
# Ensure EGL libraries are accessible (if needed)
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev

# Run normally
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/aloha_sim/compose.yml up --build
```

#### For `aloha_real` example:
```bash
# Ensure user is in device groups
sudo usermod -aG dialout,video $(whoami)
# Log out and back in for group changes to take effect

# Then run normally
docker compose -f examples/aloha_real/compose.yml up --build
```

## Troubleshooting

### Issue: `docker: command not found`

**Cause**: Environment variables not loaded

**Solution**:
```bash
source ~/.bashrc
# or restart your terminal
```

### Issue: `Cannot connect to the Docker daemon`

**Cause**: Docker daemon not running or DOCKER_HOST not set

**Solution**:
```bash
# Check if Docker is running
systemctl --user status docker.service

# Start if not running
systemctl --user start docker.service

# Set DOCKER_HOST for current session
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
```

### Issue: `GPU not accessible in containers`

**Cause**: nvidia-container-toolkit not configured properly

**Solution**:
```bash
# Check daemon.json
cat ~/.config/docker/daemon.json

# Should contain nvidia runtime configuration
# If not, re-run:
bash scripts/docker/install_nvidia_container_toolkit.sh
```

### Issue: "Operation not permitted" errors

**Cause**: Container trying to perform root-level operations

**Solution**:
1. Check if the operation is actually necessary
2. Use specific capabilities instead of `privileged: true`:
   ```yaml
   cap_add:
     - SYS_ADMIN
     - SYS_PTRACE
   ```

### Issue: X11 connection refused

**Cause**: X11 access not granted

**Solution**:
```bash
xhost +local:docker
# Or for better security:
xhost +local:$(whoami)
```

### Issue: Cannot bind to port < 1024

**Cause**: Rootless mode cannot use privileged ports

**Solution**:
- Use ports >= 1024
- Or enable unprivileged port binding:
  ```bash
  sudo sysctl -w net.ipv4.ip_unprivileged_port_start=0
  ```

### Issue: Device not accessible in container

**Cause**: User doesn't have permissions for device

**Solution**:
```bash
# Add user to appropriate groups
sudo usermod -aG dialout,video,plugdev $(whoami)
# Log out and back in for changes to take effect
```

## System Requirements Met

After running these scripts, your system will meet all openpi Docker requirements:

- ✅ Docker installed in rootless mode
- ✅ NVIDIA container toolkit installed and configured
- ✅ CDI support enabled
- ✅ No snap Docker (incompatible with nvidia-container-toolkit)
- ✅ No Docker Desktop (incompatible with NVIDIA runtime)
- ✅ GPU access working in containers
- ✅ Docker Compose V2 GPU syntax supported
- ✅ No system-level conflicts

## Next Steps

After successful installation:

1. **Test inference without a robot**: See [examples/simple_client/README.md](../../examples/simple_client/README.md)

2. **Fine-tune models**: See the main [README.md](../../README.md#fine-tuning-base-models-on-your-own-data)

3. **Run LIBERO benchmark**: See [examples/libero/README.md](../../examples/libero/README.md)

4. **Run on ALOHA robot**: See [examples/aloha_real/README.md](../../examples/aloha_real/README.md)

## Important Notes

1. **No system CUDA required**: CUDA libraries are included in Docker images. You only need NVIDIA drivers on the host.

2. **Automatic startup**: Docker will start automatically on boot (via systemd user service with linger enabled)

3. **No sudo for docker commands**: All docker commands run without sudo in rootless mode

4. **Multiple users**: Each user can run their own rootless Docker daemon independently

5. **Security**: Rootless mode provides better security isolation while maintaining full GPU access

## Support

If you encounter issues:

1. Check the [ROOTLESS_LIMITATIONS.md](ROOTLESS_LIMITATIONS.md) document
2. Run the verification script: `bash scripts/docker/verify_gpu_access.sh`
3. Check Docker logs: `journalctl --user -u docker.service -n 50`
4. Check NVIDIA drivers: `nvidia-smi`
5. File an issue at https://github.com/Physical-Intelligence/openpi/issues

## References

- [Docker Rootless Mode](https://docs.docker.com/engine/security/rootless/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [CDI Support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
- [OpenPI Documentation](../../README.md)
