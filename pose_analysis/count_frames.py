# Script to see how many frames are in the provided video file
# e.g. `python count_frames.py video.mp4` -> `1234`

import sys
import cv2


def main():

    if len(sys.argv) != 2:
        print("Usage: python count_frames.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        sys.exit(1)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)

    cap.release()


if __name__ == "__main__":
    main()
