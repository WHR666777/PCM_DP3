import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py


def decode_h5_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        pointcloud = root["/pointcloud"][()]
        if "state" in root and "vector" in root["state"] and "action" in root and "vector" in root["action"]:
            state = root["/state/vector"][()]
            action = root["/action/vector"][()]
            action_mode = None
            if "task_frame" in root and "action_mode" in root["task_frame"].attrs:
                action_mode = decode_h5_attr(root["task_frame"].attrs["action_mode"])

            if state.ndim != 2 or action.ndim != 2:
                raise ValueError(f"{dataset_path}: state/action must be 2D, got {state.shape} and {action.shape}")
            if state.shape[0] != action.shape[0]:
                raise ValueError(
                    f"{dataset_path}: state/action length mismatch: {state.shape[0]} vs {action.shape[0]}"
                )
            if pointcloud.shape[0] != state.shape[0]:
                raise ValueError(
                    f"{dataset_path}: pointcloud/state length mismatch: {pointcloud.shape[0]} vs {state.shape[0]}"
                )
            if state.shape[1] != 20 or action.shape[1] != 20:
                raise ValueError(
                    f"{dataset_path}: expected cartesian_6d state/action dim 20, got {state.shape[1]} and {action.shape[1]}"
                )
            if action_mode is not None and action_mode not in {"absolute_task_target", "local_delta", "keep"}:
                raise ValueError(f"{dataset_path}: unknown task_frame/action_mode={action_mode}")
            if action_mode == "keep":
                raise ValueError(f"{dataset_path}: action_mode=keep but /action/vector exists; metadata conflict")

            return state, action, pointcloud, "cartesian_6d", action_mode

        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        vector = root["/joint_action/vector"][()]

    return left_gripper, left_arm, right_gripper, right_arm, vector, pointcloud, "legacy_joint", None


def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = "../../data/" + str(task_name) + "/" + str(task_config)

    total_count = 0

    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    point_cloud_arrays = []
    episode_ends_arrays, state_arrays, action_arrays = [], [], []
    expected_action_mode = None

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        loaded = load_hdf5(load_path)
        # load_hdf5 returns (..., data_format, action_mode)
        data_format = loaded[-2]

        if data_format == "cartesian_6d":
            state_all, action_all, pointcloud_all, _, action_mode = loaded
            if action_mode is not None:
                if expected_action_mode is None:
                    expected_action_mode = action_mode
                elif expected_action_mode != action_mode:
                    raise ValueError(
                        f"Mixed task_frame/action_mode detected: expected {expected_action_mode}, got {action_mode} at {load_path}"
                    )
            for j in range(0, state_all.shape[0] - 1):
                point_cloud_arrays.append(pointcloud_all[j])
                state_arrays.append(state_all[j])
                action_arrays.append(action_all[j])
            episode_len = state_all.shape[0]
        else:
            (
                left_gripper_all,
                left_arm_all,
                right_gripper_all,
                right_arm_all,
                vector_all,
                pointcloud_all,
                _,
                _,
            ) = loaded

            for j in range(0, left_gripper_all.shape[0]):
                pointcloud = pointcloud_all[j]
                joint_state = vector_all[j]

                if j != left_gripper_all.shape[0] - 1:
                    point_cloud_arrays.append(pointcloud)
                    state_arrays.append(joint_state)
                if j != 0:
                    action_arrays.append(joint_state)
            episode_len = left_gripper_all.shape[0]

        current_ep += 1
        total_count += episode_len - 1
        episode_ends_arrays.append(total_count)

    print()
    try:
        episode_ends_arrays = np.array(episode_ends_arrays)
        state_arrays = np.array(state_arrays)
        point_cloud_arrays = np.array(point_cloud_arrays)
        action_arrays = np.array(action_arrays)
    
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        state_chunk_size = (100, state_arrays.shape[1])
        action_chunk_size = (100, action_arrays.shape[1])
        point_cloud_chunk_size = (100,) + point_cloud_arrays.shape[1:]
        zarr_data.create_dataset(
            "point_cloud",
            data=point_cloud_arrays,
            chunks=point_cloud_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "state",
            data=state_arrays,
            chunks=state_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "action",
            data=action_arrays,
            chunks=action_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_meta.create_dataset(
            "episode_ends",
            data=episode_ends_arrays,
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )
    except ZeroDivisionError as e:
        print("If you get a `ZeroDivisionError: division by zero`, check that `data/pointcloud` in the task config is set to true.")
        raise 
    except Exception as e:
        print(f"An unexpected error occurred ({type(e).__name__}): {e}")
        raise

if __name__ == "__main__":
    main()
