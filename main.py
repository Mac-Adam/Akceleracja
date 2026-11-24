import cv2
import numpy as np
import argparse
import os
from frame_reader import load_cap
from opencl_engine import OpenCLEngine


def parse_args():
    parser = argparse.ArgumentParser(description="OpenCV + OpenCL + MOG")
    parser.add_argument(
        "--source",
        type=int,
        default=None,
        help="Camera index (e.g., '0')",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        choices=["highway", "office", "pedestrian"],
        help="Path to an image folder (e.g., './data/frames')",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="gmm_kernel.cl",
        help="Path to the OpenCL GMM kernel file",
    )
    parser.add_argument(
        "--gmm-k",
        type=int,
        default=3,
        help="Number of Gaussians per pixel",
    )
    parser.add_argument(
        "--gmm-alpha",
        type=float,
        default=0.01,
        help="GMM learning rate in (0, 1)",
    )
    parser.add_argument(
        "--fps-limit",
        type=int,
        default=30,
        help="Delay in ms between frames (useful when reading from a folder)",
    )
    return parser.parse_args()


ESC = ord("\x1b")

proj_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_HIGHWAY = os.path.join(proj_dir, "dataset", "highway", "input", "in%06d.jpg")
DATASET_OFFICE = os.path.join(proj_dir, "dataset", "office", "input", "in%06d.jpg")
DATASET_PEDESTRIAN = os.path.join(
    proj_dir, "dataset", "pedestrian", "input", "in%06d.jpg"
)


def main():
    args = parse_args()

    if args.path == "highway":
        args.path = DATASET_HIGHWAY
    elif args.path == "office":
        args.path = DATASET_OFFICE
    elif args.path == "pedestrian":
        args.path = DATASET_PEDESTRIAN

    cap = load_cap(args.source, args.path)

    ocl_engine = OpenCLEngine(
        args.kernel,
        num_gaussians=args.gmm_k,
        alpha=args.gmm_alpha,
    )

    mog_cv2 = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )

    wait_time = 1000 // args.fps_limit if args.fps_limit > 0 else 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end). Exiting ...")
            break

        bgra_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        gmm_mask = ocl_engine.apply_gmm(bgra_frame)

        fg_mask = mog_cv2.apply(frame)

        cv2.imshow("1. Original Frame", frame)
        cv2.imshow("2. OpenCL GMM Mask (custom)", gmm_mask)
        cv2.imshow("3. MOG2 Reference Mask", fg_mask)

        if cv2.waitKey(wait_time) & 0xFF == ESC:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
