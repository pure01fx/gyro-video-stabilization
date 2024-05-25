import os
from argparse import ArgumentParser

import cv2
import imageio
import numpy as np


def calculate_psnr(img1, img2):
    mse = cv2.norm(img1, img2, cv2.NORM_L2SQR)
    mse = mse / (img1.shape[0] * img1.shape[1])
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def diff_rotation(
    gyro_timestamps: np.ndarray,
    time_range: tuple[float, float],
    anglev: np.ndarray,
):
    """
    Calculate the rotation matrix from the integral of the angular velocities read by the gyro.
    Using gyro readings in the time range [ta, tb].

    Args:
        gyro_timestamps (np.ndarray): The timestamps of the gyro readings
        time_range (tuple[float, float]): The start and end times of the time range
        anglev (np.ndarray): The angular velocities corresponding to the gyro readings

    Returns:
        np.ndarray: The rotation matrix
    """
    gyro_idx_start = np.searchsorted(gyro_timestamps, time_range[0])
    gyro_idx_end = np.searchsorted(gyro_timestamps, time_range[1]) - 1
    assert gyro_idx_start <= gyro_idx_end, "Time range leads to no gyro readings"

    R = np.eye(3)

    for i in range(gyro_idx_start, gyro_idx_end + 1):
        # Calculate the integral time $dt = tb - ta$
        start = time_range[0] if i == gyro_idx_start else gyro_timestamps[i - 1]
        end = time_range[1] if i == gyro_idx_end else gyro_timestamps[i]
        dt = end - start

        angle = dt * anglev[i - 1, :]  # Integrate the angular velocity
        theta = np.linalg.norm(angle)
        angle = angle / theta
        # Calculate the skew-symmetric matrix to replace the cross product
        skew = np.array(
            [
                [0, -angle[2], angle[1]],
                [angle[2], 0, -angle[0]],
                [-angle[1], angle[0], 0],
            ]
        )
        # Calculate the rotation matrix
        # Using the Rodrigues' rotation formula
        nnT = np.outer(angle, angle)
        R = R @ (np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * nnT + np.sin(theta) * skew)
    return R


def make_homo_seq(
    frametime: tuple[float, float],
    patch: int,
    gyro_timestamps: np.ndarray,
    anglev: np.ndarray,
    K: np.ndarray,
):
    """
    Generate a sequence of homography matrices based on gyroscopic data.

    Args:
        frametime (tuple[float, float]): A tuple representing the start and end time of taking the frame.
        patch (int): The number of patches to divide the frame into.
        gyro_timestamps (np.array): An array of gyroscopic timestamps.
        anglev (np.array): An array of angular velocities.
        K (np.array): The camera intrinsic matrix.

    Returns:
        np.array: An array of homography matrices.

    Raises:
        AssertionError: If any element in the homography matrix is NaN.
    """
    timespan = frametime[1] - frametime[0]
    K_inv = np.linalg.inv(K)
    homos = []

    for i in range(patch):
        time_range: tuple[float, float] = tuple(x + timespan * i / patch for x in frametime)  # type: ignore
        R = diff_rotation(gyro_timestamps, time_range, anglev)
        homo = K @ R @ K_inv
        homo /= homo[-1, -1]
        assert True not in np.isnan(homo)
        homos.append(homo)

    homos = np.array(homos)
    assert np.allclose(homos[:, -1, -1], 1)
    return homos


def transform_image(img: np.ndarray, homos: np.ndarray):
    """
    Transforms an image using homography matrices.
    Vertically split the img and transform rows using homography matrix.

    Args:
        img (numpy.ndarray): The input image.
        homos (numpy.ndarray): The homography matrices.

    Returns:
        numpy.ndarray: The transformed image.
    """
    H, W = img.shape[:2]
    h = H // homos.shape[0]
    out_img = np.zeros((H, W, 3), dtype=np.uint8)

    for row in range(homos.shape[0]):
        tmp = cv2.warpPerspective(img, homos[row], (W, H))
        r = slice(row * h, H if row == homos.shape[0] - 1 else ((row + 1) * h))
        out_img[r] = tmp[r]

    return out_img


def rs_correct(
    frame: cv2.typing.MatLike,
    frametime: tuple[float, float],
    gyro_timestamps: np.ndarray,
    anglev: np.ndarray,
    K: np.ndarray,
    patch: int,
):
    """
    Corrects the rolling shutter effect in a video frame using gyroscope data.

    Args:
        frame (cv2.typing.MatLike): The input video frame to be corrected.
        frametime (tuple[float, float]): The time range taking this frame.
        gyro_timestamps (np.ndarray): The timestamps of the gyroscope data.
        anglev (np.ndarray): The angular velocities from the gyroscope.
        K (np.ndarray): The camera intrinsic matrix.

    Returns:
        cv2.typing.MatLike: The corrected video frame.
    """
    homos = make_homo_seq(frametime, patch, gyro_timestamps, anglev, K)
    return transform_image(frame, homos)


### Pre-defined parameters ###
K = np.array([[573.8534, -0.6974, 406.0101], [0, 575.0448, 309.0112], [0, 0, 1]])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filename", required=True)
    parser.add_argument("--idx", nargs="+", required=True, type=int)
    parser.add_argument("--save", default=False, type=bool)
    parser.add_argument("--patch", default=10, type=int)
    args = parser.parse_args()
    file_root = args.filename
    idx = tuple(args.idx)
    patch = args.patch
    save = args.save

    def read_img(idx):
        return cv2.imread(os.path.join(file_root, f"reshape/RE_frame-{idx}.jpg"))

    os.makedirs(os.path.join(file_root, "gif"), exist_ok=True)

    frame_timestamps = np.loadtxt(os.path.join(file_root, "framestamp.txt"), dtype=np.float_)
    gyro = np.loadtxt(os.path.join(file_root, "gyro.txt"), dtype=np.float_, delimiter=",")
    gyro_timestamps = gyro[:, -1]
    anglev = gyro[:, :3]
    # Convert the angle to the camera coordinate system
    # (gyro's x is the camera's y)
    anglev[:, (0, 1)] = anglev[:, (1, 0)]

    frame0 = read_img(idx[0])
    frame1 = frame0
    psnr_gaps = []
    for i in range(idx[0], idx[1]):
        frame0 = frame1
        frame1 = read_img(i + 1)

        frametime = (frame_timestamps[i - 1], frame_timestamps[i])
        frame0_warpped = rs_correct(frame0, frametime, gyro_timestamps, anglev, K, patch)

        psnr_orig = calculate_psnr(frame0[15:-15, 15:-15], frame1[15:-15, 15:-15])
        psnr_gyro = calculate_psnr(frame0_warpped[15:-15, 15:-15], frame1[15:-15, 15:-15])
        psnr_gap = psnr_gyro - psnr_orig
        psnr_gaps.append(psnr_gap)
        print(f"{i} - {i+1}: orig={psnr_orig:.3f}, gyro={psnr_gyro:.3f}, gap={psnr_gap:.3f}")

        if save:
            gif_frame0 = np.concatenate([frame0, frame0], axis=1)
            gif_frame1 = np.concatenate([frame1, frame0_warpped], axis=1)
            gif_file = os.path.join(file_root, "gif", f"frame-{i}-{psnr_gyro - psnr_orig:.2f}.gif")
            imageio.mimsave(gif_file, [gif_frame0, gif_frame1], duration=1.0, loop=0)

    print(f"Average PSNR gap: {np.mean(psnr_gaps):.3f}")
