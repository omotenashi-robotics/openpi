#!/bin/bash
# Wrapper script to run ALOHA sim with GPU support
# Docker Compose v2.39.2 has a bug where GPU passthrough doesn't work with 'up' command
# This script uses docker run directly which works correctly
#
# Usage:
#   ./run_aloha_sim.sh              # Default: ALOHA_SIM mode (sim checkpoint + runtime container)
#   ./run_aloha_sim.sh towel        # Towel mode (towel checkpoint, client runs on host)

set -e

cd "$(dirname "$0")"

# Parse mode argument
MODE="${1:-aloha_sim}"

if [[ "$MODE" != "aloha_sim" && "$MODE" != "towel" ]]; then
  echo "Error: Invalid mode '$MODE'"
  echo "Usage: $0 [aloha_sim|towel]"
  echo "  aloha_sim (default): Run ALOHA simulation with sim checkpoint"
  echo "  towel: Run server with towel checkpoint (client runs on host)"
  exit 1
fi

echo "Running in $MODE mode..."

# Build images if needed
echo "Building Docker images..."
docker build -t openpi_server -f scripts/docker/serve_policy.Dockerfile .

# Only build runtime image for aloha_sim mode
if [[ "$MODE" == "aloha_sim" ]]; then
  docker build -t aloha_sim -f examples/aloha_sim/Dockerfile .
fi

# Start openpi_server in background
# Note: We override the CMD to use PYTHONPATH instead of 'uv run' to avoid
# package reinstallation that breaks JAX CUDA support
echo "Starting openpi_server..."

# Set server command based on mode
if [[ "$MODE" == "aloha_sim" ]]; then
  SERVER_CMD="cd /app && PYTHONPATH=/app/src:/.venv/lib/python3.11/site-packages /.venv/bin/python scripts/serve_policy.py --env ALOHA_SIM"
  echo "  Using ALOHA_SIM checkpoint"
elif [[ "$MODE" == "towel" ]]; then
  SERVER_CMD="cd /app && PYTHONPATH=/app/src:/.venv/lib/python3.11/site-packages /.venv/bin/python scripts/serve_policy.py policy:checkpoint --policy.config=pi0_aloha_towel --policy.dir=gs://openpi-assets/checkpoints/pi0_aloha_towel --default_prompt='fold the towel'"
  echo "  Using towel checkpoint (pi0_aloha_towel)"
fi

docker run -d \
  --name aloha_openpi_server \
  --gpus device=0 \
  --network host \
  -e OPENPI_DATA_HOME=/openpi_assets \
  -e IS_DOCKER=true \
  -v "$(pwd):/app" \
  -v "${HOME}/.cache/openpi:/openpi_assets" \
  openpi_server \
  /bin/bash -c "$SERVER_CMD"

# Wait for server to start
echo "Waiting for server to start..."
sleep 5

if [[ "$MODE" == "aloha_sim" ]]; then
  # Start runtime container for aloha_sim mode
  echo "Starting runtime container..."
  docker run -it --rm \
    --name aloha_runtime \
    --network host \
    --privileged \
    -v "$(pwd):/app" \
    -v "$(pwd)/data:/data" \
    aloha_sim

  # Cleanup: stop server when runtime exits
  echo "Stopping openpi_server..."
  docker stop aloha_openpi_server
  docker rm aloha_openpi_server
elif [[ "$MODE" == "towel" ]]; then
  # For towel mode, just keep server running and show instructions
  echo ""
  echo "=========================================="
  echo "OpenPI Server is running with towel checkpoint!"
  echo "=========================================="
  echo ""
  echo "To run the towel client on your host machine:"
  echo ""
  echo "  # Install gym_aloha if not already installed"
  echo "  uv pip install gym-aloha"
  echo ""
  echo "  # Run the towel client"
  echo "  python examples/simple_client/towel_client.py \\"
  echo "    --use-sim \\"
  echo "    --prompt 'fold the towel' \\"
  echo "    --num-steps 20"
  echo ""
  echo "Press Ctrl+C to stop the server when done."
  echo "=========================================="
  echo ""

  # Keep server running until user interrupts
  trap "echo 'Stopping server...'; docker stop aloha_openpi_server; docker rm aloha_openpi_server; exit 0" INT TERM

  # Monitor server logs
  docker logs -f aloha_openpi_server
fi
