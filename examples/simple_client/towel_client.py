"""
Modified client for testing the towel checkpoint with simulation-based observations.

This script runs the ALOHA simulation and sends actual observations to the towel checkpoint,
instead of random noise. This provides more realistic testing.
"""
import dataclasses
import logging
import pathlib
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

try:
    import gymnasium
    import gym_aloha
    SIM_AVAILABLE = True
except ImportError:
    SIM_AVAILABLE = False
    logging.warning("gym_aloha not available. Install with: uv pip install gym-aloha")

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "localhost"
    port: int = 8000

    # Number of steps to run the policy for.
    num_steps: int = 20

    # Task to run in simulation (if using sim observations)
    sim_task: str = "gym_aloha/AlohaTransferCube-v0"

    # Whether to use simulation observations (requires gym_aloha)
    use_sim: bool = True

    # Prompt to send with observations
    prompt: str = "fold the towel"

    # Path to save video (optional)
    video_path: pathlib.Path | None = None


class AlohaSimObservationGenerator:
    """Generate observations from ALOHA simulation."""

    def __init__(self, task: str = "gym_aloha/AlohaTransferCube-v0"):
        if not SIM_AVAILABLE:
            raise ImportError(
                "gym_aloha is not available. Install with: uv pip install gym-aloha"
            )

        self.env = gymnasium.make(task, obs_type="pixels_agent_pos")
        self.reset()

    def reset(self):
        """Reset the environment."""
        gym_obs, _ = self.env.reset()
        self.last_obs = self._convert_observation(gym_obs)
        return self.last_obs

    def step(self, action: np.ndarray):
        """Step the environment and return observation."""
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        gym_obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = self._convert_observation(gym_obs)

        if terminated or truncated:
            self.reset()

        return self.last_obs

    def _convert_observation(self, gym_obs: dict) -> dict:
        """Convert gym observation to format expected by towel checkpoint.

        The towel checkpoint expects:
        - state: [14] joint positions (left_arm[6], left_gripper[1], right_arm[6], right_gripper[1])
        - images: dict with keys cam_high, cam_low, cam_left_wrist, cam_right_wrist
                  Each image is [3, 224, 224] uint8
        """
        # Get the top camera image
        img_top = gym_obs["pixels"]["top"]  # [H, W, 3]

        # Resize and convert to uint8 if needed
        img_top = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_top, 224, 224))

        # Convert to [C, H, W] format
        img_top = np.transpose(img_top, (2, 0, 1))

        # Get state (agent_pos is 14-dimensional for ALOHA)
        state = gym_obs["agent_pos"]

        # The simulation only provides 1 camera (top view).
        # For the towel checkpoint which expects 4 cameras, we'll:
        # 1. Use the top camera as cam_high
        # 2. Create slightly modified versions for other views (since we don't have real multi-camera)
        # 3. Alternatively, use black images for missing cameras

        # Option 1: Use the same image for all cameras (not ideal but better than random)
        images = {
            "cam_high": img_top,
            "cam_low": img_top,  # Reuse top camera
            "cam_left_wrist": img_top,  # Reuse top camera
            "cam_right_wrist": img_top,  # Reuse top camera
        }

        # Option 2: Use black images for missing cameras (uncomment if you prefer)
        # images = {
        #     "cam_high": img_top,
        #     "cam_low": np.zeros_like(img_top),
        #     "cam_left_wrist": np.zeros_like(img_top),
        #     "cam_right_wrist": np.zeros_like(img_top),
        # }

        return {
            "state": state,
            "images": images,
        }

    def get_observation(self, prompt: str = "fold the towel") -> dict:
        """Get current observation with prompt."""
        obs = self.last_obs.copy()
        obs["prompt"] = prompt
        return obs


class StaticObservationGenerator:
    """Generate static 'neutral' observations for ALOHA (fallback when sim not available)."""

    def __init__(self):
        # Neutral joint positions (roughly at rest pose)
        self.state = np.array([0.0, -1.5, 1.5, 0.0, 0.0, 0.0, 0.0,  # left arm + gripper
                               0.0, -1.5, 1.5, 0.0, 0.0, 0.0, 0.0]) # right arm + gripper

        # Create a simple gradient image instead of random noise
        self.base_image = self._create_gradient_image()

    def _create_gradient_image(self) -> np.ndarray:
        """Create a simple gradient image (better than random)."""
        img = np.zeros((3, 224, 224), dtype=np.uint8)

        # Create a gradient from top to bottom
        for i in range(224):
            img[:, i, :] = int(255 * i / 224)

        return img

    def get_observation(self, prompt: str = "fold the towel") -> dict:
        """Get static observation with prompt."""
        return {
            "state": self.state.copy(),
            "images": {
                "cam_high": self.base_image.copy(),
                "cam_low": self.base_image.copy(),
                "cam_left_wrist": self.base_image.copy(),
                "cam_right_wrist": self.base_image.copy(),
            },
            "prompt": prompt,
        }

    def step(self, action: np.ndarray):
        """No-op for static generator."""
        return self.get_observation()

    def reset(self):
        """No-op for static generator."""
        return self.get_observation()


def main(args: Args) -> None:
    """Main function to run the towel client."""

    # Create observation generator
    if args.use_sim and SIM_AVAILABLE:
        logger.info("Using ALOHA simulation for observations")
        obs_generator = AlohaSimObservationGenerator(task=args.sim_task)
    else:
        if args.use_sim and not SIM_AVAILABLE:
            logger.warning("Simulation requested but gym_aloha not available. Using static observations.")
        else:
            logger.info("Using static observations")
        obs_generator = StaticObservationGenerator()

    # Create policy client
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    # Warm up the model
    logger.info("Warming up model...")
    for _ in range(2):
        obs = obs_generator.get_observation(prompt=args.prompt)
        policy.infer(obs)

    # Run inference loop
    logger.info(f"Running {args.num_steps} inference steps...")
    frames = []

    for step in range(args.num_steps):
        # Get observation
        obs = obs_generator.get_observation(prompt=args.prompt)

        # Run inference
        start_time = time.time()
        result = policy.infer(obs)
        inference_time = time.time() - start_time

        # Get action
        action = result["actions"]
        logger.info(
            f"Step {step+1}/{args.num_steps}: "
            f"Action shape: {action.shape}, "
            f"Inference time: {inference_time*1000:.1f}ms"
        )

        # Apply action to get next observation (if using sim)
        if args.use_sim and SIM_AVAILABLE:
            # Use the first action from the chunk
            next_obs = obs_generator.step(action[0])

            # Save frame if video path specified
            if args.video_path is not None:
                frame = obs["images"]["cam_high"]  # [C, H, W]
                frame = np.transpose(frame, (1, 2, 0))  # [H, W, C]
                frames.append(frame)

    # Save video if requested
    if args.video_path is not None and frames:
        logger.info(f"Saving video to {args.video_path}")
        save_video(frames, args.video_path)

    logger.info("Done!")


def save_video(frames: list, output_path: pathlib.Path) -> None:
    """Save frames as a video."""
    try:
        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (width, height))

        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        logger.info(f"Video saved to {output_path}")
    except ImportError:
        logger.warning("OpenCV not available. Cannot save video.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(tyro.cli(Args))
