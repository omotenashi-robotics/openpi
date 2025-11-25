"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
import json
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    cameras: list[str],
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def _resize_to_chw(img: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """Ensure image is CHW and resized to target dims."""
    import cv2

    # Normalize to HWC for resize.
    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {img.shape}")
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img_hwc = np.transpose(img, (1, 2, 0))
    else:
        img_hwc = img
    if img_hwc.shape[-1] != 3:
        raise ValueError(f"Expected image channel size 3, got {img_hwc.shape}")

    resized = cv2.resize(img_hwc, (width, height), interpolation=cv2.INTER_AREA)
    return np.transpose(resized, (2, 0, 1))  # CHW

def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        # Normalize each frame to CHW 480x640 to match expected feature shape.
        imgs_array = np.array([
            _resize_to_chw(frame)
            if frame.shape == (480, 848, 3)
            else frame  
            for frame in imgs_array
        ])

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
    cameras: list[str],
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:]).float()
        action = torch.from_numpy(ep["/action"][:]).float()

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:]).float()

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:]).float()

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            cameras,
        )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    cameras: list[str],
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path, cameras)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }
            frame["task"] = task
            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def load_dataset_append_only(
    repo_id: str,
    repo_root: Path,
    dataset_config: DatasetConfig,
) -> LeRobotDataset:
    """Load existing dataset without scanning data files, using meta only."""
    obj = LeRobotDataset.__new__(LeRobotDataset)
    obj.meta = LeRobotDatasetMetadata(repo_id, root=repo_root)
    obj.repo_id = obj.meta.repo_id
    obj.root = obj.meta.root
    obj.revision = None
    obj.tolerance_s = dataset_config.tolerance_s
    obj.image_writer = None
    obj.episode_buffer = None
    obj.episodes = None
    obj.image_transforms = None
    obj.delta_timestamps = None
    obj.delta_indices = None
    obj.video_backend = dataset_config.video_backend
    obj.hf_dataset = obj.create_hf_dataset()
    obj.episode_data_index = None
    return obj


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    resume: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    repo_root = HF_LEROBOT_HOME / repo_id
    data_root = repo_root / "data"

    if (repo_root.exists()) and not resume:
        shutil.rmtree(repo_root)

    if not raw_dir.exists():
        raise ValueError(f"raw_dir does not exist: {raw_dir}")

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        raise ValueError(f"No episode_*.hdf5 files found under {raw_dir}")

    cameras = get_cameras(hdf5_files)

    if resume and repo_root.exists():
        dataset = load_dataset_append_only(
            repo_id=repo_id,
            repo_root=repo_root,
            dataset_config=dataset_config,
        )
        dataset.start_image_writer(dataset_config.image_writer_processes, dataset_config.image_writer_threads)
    else:
        dataset = create_empty_dataset(
            repo_id,
            robot_type="mobile_aloha" if is_mobile else "aloha",
            cameras=cameras,
            mode=mode,
            has_effort=has_effort(hdf5_files),
            has_velocity=has_velocity(hdf5_files),
            dataset_config=dataset_config,
        )

    episodes_all = episodes if episodes is not None else list(range(len(hdf5_files)))
    if resume and (repo_root / "meta" / "episodes.jsonl").exists():
        existing_task_count = 0
        episodes_meta = repo_root / "meta" / "episodes.jsonl"
        with episodes_meta.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if task in rec.get("tasks", []):
                    existing_task_count += 1
        episodes_to_run = episodes_all[existing_task_count:]
        print(
            f"[resume] Task '{task}' already has {existing_task_count} episodes; will process {len(episodes_to_run)} more."
        )
    else:
        episodes_to_run = episodes_all

    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        cameras=cameras,
        episodes=episodes_to_run,
    )
    # dataset.consolidate()

    saved_episode_files = list(data_root.glob("**/episode_*.parquet"))
    expected_episodes = len(episodes_all)
    if len(saved_episode_files) != expected_episodes:
        print(
            f"Dataset incomplete: saved {len(saved_episode_files)} / {expected_episodes} episodes. "
            "The conversion may have been interrupted (e.g., OOM or kill)."
        )

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
