#!/usr/bin/env python3
"""
Local Pose Analyzer

This script runs the pose analysis pipeline locally instead of in AWS Lambda.
It reuses the core processing functions from the main.py and pose_detection.py files.

Usage:
    python local_pose_analyzer.py --video <path_to_video> --output <output_directory> --joints left_knee right_knee --frame_step 10
"""

import os
import sys
import warnings
import logging

# Set environment variables before importing any other libraries
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
# Additional environment variable to suppress MediaPipe warnings
os.environ["GLOG_minloglevel"] = "2"  # Suppress glog messages (used by MediaPipe)

# Set up logging to filter out unwanted messages
logging.basicConfig(level=logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)
# Suppress MediaPipe's root logger as well
logging.getLogger().setLevel(logging.ERROR)

# Configure warnings filter (must be done before imports)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*feedback tensors.*")
warnings.filterwarnings("ignore", message=".*Failed to set name for thread.*")
# Add more specific filter for the inference feedback manager warning
warnings.filterwarnings("ignore", message=".*inference_feedback_manager.*")
warnings.filterwarnings("ignore", message=".*Feedback manager requires.*")

# Now import the rest of your modules
import json
import argparse
import time
import datetime
import cv2 as OpenCV

# Import the pose_detection module
import pose_detection as pose

# Import core functions from main module
from main import process_video, draw_leg_landmarks, estimate_pose, add_frame_labels


def run_local_analysis(
    video_path="./sample_video.mp4", output_dir="./output", frame_step=10, joints=None
):
    """
    Run pose analysis on a local video file and save results to a local directory.

    Args:
        video_path (str): Path to the input video file
        output_dir (str): Path to the output directory
        frame_step (int): Number of frames to skip between analysis
        joints (list): List of joints to analyze, e.g. ["left_knee", "right_knee"]

    Returns:
        dict: Dictionary containing joint angle measurements
    """
    start_time = time.time()

    if joints is None:
        joints = ["left_knee"]

    print(f"Starting analysis of: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Frame step: {frame_step}")
    print(f"Joints to analyze: {joints}")

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process the video using the same function from the Lambda version
    est_angles = process_video(video_path, output_dir, frame_step, joints)

    # Save the estimated angles to a JSON file
    with open(f"{output_dir}/estimations.json", "w") as outfile:
        json.dump(est_angles, outfile, indent=4)

    # Calculate and display execution time
    duration = time.time() - start_time
    formatted_duration = str(datetime.timedelta(seconds=int(duration)))

    print("\nAnalysis Complete:")
    print(f"  Duration: {formatted_duration} ({duration:.2f} seconds)")
    print(f"  Results saved to: {output_dir}")

    # Summary of results
    print("\nJoint Angle Summary:")
    for joint_name, angles in est_angles.items():
        if angles:
            min_angle = min(angles)
            max_angle = max(angles)
            avg_angle = sum(angles) / len(angles)
            print(f"  {joint_name}:")
            print(f"    - Min: {min_angle:.1f}°")
            print(f"    - Max: {max_angle:.1f}°")
            print(f"    - Avg: {avg_angle:.1f}°")
            print(f"    - Measurements: {len(angles)}")

    return est_angles


def main():
    """Parse command line arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="Local Pose Analysis Tool")
    parser.add_argument(
        "--video",
        type=str,
        required=False,
        default="./sample_video.mp4",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="./output",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        required=False,
        default=10,
        help="Number of frames to analyze (total frames / frame_step)",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        required=False,
        default=["left_knee"],
        help="Joints to analyze (e.g., left_knee right_knee)",
    )

    args = parser.parse_args()

    # Run the analysis with the provided arguments
    run_local_analysis(
        video_path=args.video,
        output_dir=args.output,
        frame_step=args.frame_step,
        joints=args.joints,
    )


if __name__ == "__main__":
    main()
