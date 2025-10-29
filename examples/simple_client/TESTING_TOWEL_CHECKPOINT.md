# Testing the ALOHA Towel Checkpoint

## Overview

This guide shows you how to test the pre-trained towel checkpoint (`gs://openpi-assets/checkpoints/pi0_aloha_towel`) using **real simulation observations** instead of random noise.

## Option 1: Test with ALOHA Simulation Observations (Recommended)

This approach runs the ALOHA simulation and feeds actual observations to the towel checkpoint.

**Note:** The client defaults to `host=localhost` assuming the server runs on the same machine. If your server is on a different machine, use `--host <server-ip>`.

### Terminal 1: Start the Policy Server

```bash
cd /home/on-premise/workspaces/ash_ws/src/openpi

# Serve the towel checkpoint
uv run scripts/serve_policy.py \
    --default_prompt="fold the towel" \
    policy:checkpoint \
    --policy.config=pi0_aloha_towel \
    --policy.dir=gs://openpi-assets/checkpoints/pi0_aloha_towel
```

### Terminal 2: Run the Custom Towel Client

```bash
cd /home/on-premise/workspaces/ash_ws/src/openpi

# Install gym_aloha if not already installed
uv pip install gym-aloha

# Run with simulation observations
MUJOCO_GL=osmesa \
python examples/simple_client/towel_client.py \
    --use-sim \
    --prompt "fold the towel" \
    --num-steps 20
```

### What This Does

The custom `towel_client.py`:
1. ✅ **Runs actual ALOHA simulation** (MuJoCo environment)
2. ✅ **Generates real observations** with actual state and camera images
3. ✅ **Sends observations to towel checkpoint** for inference
4. ✅ **Applies predicted actions back to simulation** for realistic testing
5. ✅ **Logs inference times and action outputs**

### Optional: Save Video Output

```bash
MUJOCO_GL=osmesa
python examples/simple_client/towel_client.py \
    --use-sim \
    --prompt "fold the towel" \
    --num-steps 100 \
    --video-path outputs/towel_test.mp4
```

---

## Option 2: Test with Static Observations (Fallback)

If you can't install `gym_aloha` or want a quick test without simulation:

```bash
cd /home/on-premise/workspaces/ash_ws/src/openpi

# Run with static observations (no simulation needed)
MUJOCO_GL=osmesa
python examples/simple_client/towel_client.py \
    --no-use-sim \
    --prompt "fold the towel" \
    --num-steps 20
```

This uses static "neutral pose" observations instead of random noise - better than pure random but not as realistic as actual simulation.

---

## Option 3: Test with Random Observations (Original Simple Client)

If you just want to verify the checkpoint loads correctly:

### Terminal 1: Start the Policy Server

```bash
cd /home/on-premise/workspaces/ash_ws/src/openpi

uv run scripts/serve_policy.py 
    --default_prompt="fold the towel" \
    policy:checkpoint \
    --policy.config=pi0_aloha_towel \
    --policy.dir=gs://openpi-assets/checkpoints/pi0_aloha_towel \
```

### Terminal 2: Run the Original Simple Client

```bash
cd /home/on-premise/workspaces/ash_ws/src/openpi

# Run with random ALOHA observations
uv run examples/simple_client/main.py --env ALOHA --num-steps 20
```

#### Expected Output
```
INFO:__main__:Server metadata: {'reset_pose': [0, -1.5, 1.5, 0, 0, 0]}
Running policy: 100%|███████████████████████████████████████████| 20/20 [00:30<00:00,  1.53s/it]
                                       Timing Statistics                                       
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Metric               ┃   Mean ┃   Std ┃    P25 ┃    P50 ┃    P75 ┃    P90 ┃    P95 ┃    P99 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ client_infer_ms      │ 1530.0 │ 165.1 │ 1456.4 │ 1528.7 │ 1678.1 │ 1699.5 │ 1724.0 │ 1801.4 │
│ policy_infer_ms      │ 1264.9 │ 115.1 │ 1223.8 │ 1236.4 │ 1367.1 │ 1375.8 │ 1417.1 │ 1491.4 │
│ server_infer_ms      │ 1524.1 │ 162.9 │ 1454.0 │ 1514.1 │ 1671.5 │ 1679.6 │ 1701.2 │ 1794.9 │
│ server_prev_total_ms │ 1518.2 │ 161.3 │ 1455.0 │ 1515.5 │ 1657.9 │ 1684.0 │ 1724.1 │ 1801.5 │
└──────────────────────┴────────┴───────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

---

## Understanding the Observations

### What the Towel Checkpoint Expects

The `pi0_aloha_towel` checkpoint was trained on real ALOHA robot data and expects:

- **State**: 14-dimensional joint positions
  - `[left_arm_joints[6], left_gripper[1], right_arm_joints[6], right_gripper[1]]`

- **Images**: 4 camera views, each `[3, 224, 224]` uint8
  - `cam_high`: Top-down view
  - `cam_low`: Lower angle view
  - `cam_left_wrist`: Left wrist camera
  - `cam_right_wrist`: Right wrist camera

- **Prompt**: Text instruction (e.g., "fold the towel")

### Simulation Limitations

⚠️ **Camera Mismatch**: ALOHA simulation only provides 1 camera (`top` view), but the towel checkpoint expects 4 cameras.

**Solution in `towel_client.py`**:
We create transformed versions of the single camera to simulate different viewpoints:
- `cam_high`: Original top view (unchanged)
- `cam_low`: Vertically flipped (simulates lower viewing angle)
- `cam_left_wrist`: Horizontally flipped (simulates left side perspective)
- `cam_right_wrist`: Both axes flipped + brightness adjustment (simulates right side perspective)

This provides more diverse visual information than just duplicating the same image 4 times.

⚠️ **Task Mismatch**: The towel checkpoint was trained for folding towels, but the default simulation task is "transfer cube". The policy may not perform optimally.

⚠️ **Domain Gap**: Real-world images vs simulated images have visual differences that affect policy performance.

---

## Expected Output

When running the custom towel client, you'll see:

```
2025-10-27 14:30:00 - INFO - Using ALOHA simulation for observations
2025-10-27 14:30:01 - INFO - Server metadata: {...}
2025-10-27 14:30:01 - INFO - Warming up model...
2025-10-27 14:30:05 - INFO - Running 20 inference steps...
2025-10-27 14:30:06 - INFO - Step 1/20: Action shape: (25, 14), Inference time: 150.3ms
2025-10-27 14:30:06 - INFO - Step 2/20: Action shape: (25, 14), Inference time: 145.7ms
...
2025-10-27 14:30:15 - INFO - Done!
```

The action shape `(25, 14)` means:
- **25**: Action horizon (25 steps predicted at once)
- **14**: Action dimensions (joint positions for both arms + grippers)

---

## Comparison: Three Approaches

| Approach | Realism | Setup | Use Case |
|----------|---------|-------|----------|
| **Towel Client (Sim)** | ✅✅✅ Real observations | Requires `gym_aloha` | Best for realistic testing |
| **Towel Client (Static)** | ✅✅ Neutral pose | No dependencies | Quick verification |
| **Simple Client (Random)** | ✅ Random noise | No dependencies | Checkpoint loading test |

---

## Network Configuration

### Host Settings Explained

#### Server (Policy Server)
```bash
# Server listens on ALL network interfaces
uv run scripts/serve_policy.py ... # uses host="0.0.0.0" internally
```
- `0.0.0.0` means the server accepts connections from:
  - ✅ Localhost (`127.0.0.1`)
  - ✅ Other machines on the network
  - ✅ Docker containers

#### Client (Towel Client)
```bash
# Default: Connect to server on same machine
python examples/simple_client/towel_client.py  # uses host="localhost" by default

# Custom: Connect to server on different machine
python examples/simple_client/towel_client.py --host 192.168.1.100
```

| Scenario | Client Host Value | Use Case |
|----------|-------------------|----------|
| Same machine | `localhost` or `127.0.0.1` | Default (most common) |
| Different machine | `<server-ip>` (e.g., `192.168.1.100`) | Remote inference |
| Docker container | `host.docker.internal` or bridge IP | Running in Docker |

---

## Troubleshooting

### `gym_aloha` installation fails

```bash
cd /home/on-premise/workspaces/ash_ws/src/openpi

# Try installing in the main environment
uv pip install gym-aloha

# Or create a separate venv for simulation
uv venv --python 3.10 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
uv pip sync examples/aloha_sim/requirements.txt
uv pip install -e packages/openpi-client
```

### EGL/rendering errors

```bash
# Install EGL dependencies
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev

# Use EGL rendering
MUJOCO_GL=egl python examples/simple_client/towel_client.py --use-sim
```

### Server connection fails

- Verify the policy server is running in Terminal 1
- Check both terminals use the same host/port (default: `localhost:8000`)
- Wait a few seconds for the server to fully initialize
- If server is on a different machine, use: `--host <server-ip-address>`

### Checkpoint download is slow

The checkpoint will be automatically downloaded from `gs://openpi-assets` and cached in `~/.cache/openpi` on first use.

---

## Next Steps

### For More Realistic Testing

To properly test the towel checkpoint in simulation, you would need:

1. **Multi-camera simulation**: Modify the ALOHA simulation to provide 4 camera views
2. **Towel folding task**: Create a custom MuJoCo scene with a towel object
3. **Domain adaptation**: Fine-tune on simulation data (but you mentioned training is not possible)

### Using the Existing Sim Checkpoint

For actual deployment in the ALOHA simulation, use the checkpoint that's already trained for simulation:

```bash
# Terminal 1: Policy server with sim checkpoint
uv run scripts/serve_policy.py --env ALOHA_SIM

# Terminal 2: Run simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

This checkpoint is optimized for the simulation environment and will perform better than the towel checkpoint.

---

## References

- [OpenPI Documentation](../../README.md)
- [ALOHA Real Robot Example](examples/aloha_real/README.md)
- [ALOHA Simulation Example](examples/aloha_sim/README.md)
- [Simple Client Example](examples/simple_client/README.md)
- [Remote Inference Guide](docs/remote_inference.md)
- [Custom Towel Client Source](examples/simple_client/towel_client.py)
