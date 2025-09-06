#!/usr/bin/env python3
"""
Face Masking Tool

A simple script that takes a single image as input and outputs an image with the face/head area masked with a black rectangle.

Usage:
    python mask_face.py input_image.jpg output_image.jpg
    python mask_face.py --input input_image.jpg --output output_image.jpg
"""

import os
import sys
import argparse
import cv2 as cv
import mediapipe as mp
import numpy as np


class FaceMasker:
    def __init__(self):
        """Initialize the face masker with MediaPipe pose detection."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

        # Landmark mapping (same as in pose_detection.py)
        self.landmark_names = {
            self.mp_pose.PoseLandmark.NOSE: "NOSE",
            self.mp_pose.PoseLandmark.LEFT_EAR: "LEFT_EAR",
            self.mp_pose.PoseLandmark.RIGHT_EAR: "RIGHT_EAR",
            self.mp_pose.PoseLandmark.LEFT_SHOULDER: "LEFT_SHOULDER",
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER: "RIGHT_SHOULDER",
        }

    def mask_face(self, image):
        """
        Apply face masking to an image.

        Args:
            image (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Image with face masked
        """
        h, w, c = image.shape

        # Process the image to detect poses
        results = self.pose.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            print("No pose landmarks detected. Returning original image.")
            return image

        # Find head landmarks
        head_top = None
        head_bottom = None
        head_left = None
        head_right = None

        head_landmarks_found = []

        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if i not in self.landmark_names:
                continue

            cx, cy = int(landmark.x * w), int(landmark.y * h)
            landmark_name = self.landmark_names[i]

            if landmark_name in [
                "LEFT_EAR",
                "RIGHT_EAR",
                "NOSE",
                "LEFT_SHOULDER",
                "RIGHT_SHOULDER",
            ]:
                head_landmarks_found.append(landmark_name)

                if head_top is None or cy < head_top:
                    head_top = cy
                if head_bottom is None or cy > head_bottom:
                    head_bottom = cy
                if head_left is None or cx < head_left:
                    head_left = cx
                if head_right is None or cx > head_right:
                    head_right = cx

        if not head_landmarks_found:
            print("No head landmarks found. Returning original image.")
            return image

        print(f"Found head landmarks: {head_landmarks_found}")

        # Apply head masking
        if all(
            coord is not None
            for coord in [head_top, head_bottom, head_left, head_right]
        ):
            # Adjust head_top to include more of the head
            height = head_bottom - head_top
            head_top -= int(height * 0.3)

            # Add padding
            y_padding = 20
            x_padding = 50
            head_top = max(0, head_top - y_padding)
            head_bottom = min(h, head_bottom + y_padding)
            head_left = max(0, head_left - x_padding)
            head_right = min(w, head_right + x_padding)

            print(
                f"Masking rectangle: ({head_left}, {head_top}) to ({head_right}, {head_bottom})"
            )

            # Draw the black rectangle to mask the face
            if head_right > head_left and head_bottom > head_top:
                cv.rectangle(
                    image,
                    (head_left, head_top),
                    (head_right, head_bottom),
                    (0, 0, 0),  # Black color
                    -1,  # Filled rectangle
                )
                print("Face masking applied successfully!")
            else:
                print("Invalid rectangle coordinates. Skipping masking.")
        else:
            print("Could not determine head boundaries. Skipping masking.")

        return image

    def process_image(self, input_path, output_path):
        """
        Process a single image file.

        Args:
            input_path (str): Path to input image
            output_path (str): Path to save masked image
        """
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            return False

        # Read the image
        image = cv.imread(input_path)
        if image is None:
            print(f"Error: Could not read image: {input_path}")
            return False

        print(f"Processing image: {input_path}")
        print(f"Image dimensions: {image.shape}")

        # Apply face masking
        masked_image = self.mask_face(image)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save the result
        success = cv.imwrite(output_path, masked_image)
        if success:
            print(f"Masked image saved to: {output_path}")
            return True
        else:
            print(f"Error: Could not save image to: {output_path}")
            return False


def main():
    """Parse command line arguments and process the image."""
    parser = argparse.ArgumentParser(
        description="Mask faces in images using pose detection"
    )

    # Support both positional and named arguments
    parser.add_argument("input", nargs="?", help="Input image path")
    parser.add_argument("output", nargs="?", help="Output image path")
    parser.add_argument(
        "--input",
        "-i",
        dest="input_file",
        help="Input image path (alternative to positional argument)",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_file",
        help="Output image path (alternative to positional argument)",
    )

    args = parser.parse_args()

    # Determine input and output paths
    input_path = args.input or args.input_file
    output_path = args.output or args.output_file

    if not input_path or not output_path:
        print("Error: Both input and output paths are required.")
        print("\nUsage examples:")
        print("  python mask_face.py input.jpg output.jpg")
        print("  python mask_face.py --input input.jpg --output output.jpg")
        sys.exit(1)

    # Process the image
    masker = FaceMasker()
    success = masker.process_image(input_path, output_path)

    if success:
        print("✅ Face masking completed successfully!")
    else:
        print("❌ Face masking failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
