import json
import os
import cv2 as OpenCV
from typing import List, Dict
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import warnings
import logging

# Suppress MediaPipe warnings
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
logging.getLogger("absl").setLevel(logging.ERROR)

# Import pose.py which is in the same directory as main.py
import pose_detection as pose

s3 = boto3.client("s3")


def lambda_handler(event, context):
    print(f"event: {event}")
    # get the body of the event
    body = json.loads(event["body"])
    bucket = "roma-t.user-files"
    userId = body["userId"]
    reportId = body["reportId"]
    frame_step = body.get("frame_step", 10)
    joints = body.get("joints", ["left_knee"])

    video_key = f"{userId}/{reportId}/video.mp4"
    tmpDirectoryPath = f"/tmp/{userId}/{reportId}"
    s3OutputPath = f"{userId}/{reportId}"

    # Create output directory
    os.makedirs(tmpDirectoryPath, exist_ok=True)

    print(f"bucket: {bucket}, video_key: {video_key}")

    video_file_path = f"{tmpDirectoryPath}/video.mp4"

    try:
        s3.download_file(bucket, video_key, video_file_path)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"Error: {video_key} not found in {bucket}")
            return {
                "statusCode": 404,
                "body": json.dumps(f"Video not found: {video_key}"),
            }
        else:
            print(f"ClientError occurred: {e}")
            return {
                "statusCode": 500,
                "body": json.dumps("An error occurred while downloading the video."),
            }

    print(f"video_file_path: {video_file_path}")

    # Check file size
    file_size = os.path.getsize(video_file_path)
    print(f"file_size: {file_size}")
    if file_size == 0:
        return {"statusCode": 400, "body": json.dumps("File is empty")}

    print(f"joints: {joints}")

    est_angles = process_video(video_file_path, tmpDirectoryPath, frame_step, joints)

    with open(f"{tmpDirectoryPath}/estimations.json", "w") as outfile:
        json.dump(est_angles, outfile, indent=4)

    try:
        s3.upload_file(
            f"{tmpDirectoryPath}/estimations.json",
            bucket,
            f"{s3OutputPath}/estimations.json",
        )
        print(f"Uploaded to S3: {s3OutputPath}/estimations.json")

        # Upload landmark video if it exists
        landmark_video_path = f"{tmpDirectoryPath}/landmark_video.mp4"
        if os.path.exists(landmark_video_path):
            s3.upload_file(
                landmark_video_path,
                bucket,
                f"{s3OutputPath}/landmark_video.mp4",
            )
            print(f"Uploaded to S3: {s3OutputPath}/landmark_video.mp4")

        # Upload min/max images
        for image_file in [
            "minPlain.png",
            "minImage.png",
            "maxPlain.png",
            "maxImage.png",
        ]:
            file_path = f"{tmpDirectoryPath}/{image_file}"
            if os.path.exists(file_path):
                s3.upload_file(file_path, bucket, f"{s3OutputPath}/{image_file}")
                print(f"Uploaded to S3: {s3OutputPath}/{image_file}")

    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps("An error occurred while uploading files to S3."),
        }

    return {"statusCode": 200, "body": json.dumps(est_angles)}


def process_video(video_file_path, output_dir_path, frame_step: int, angles: List[str]):
    print(f"video_file_path: {video_file_path}")
    print(f"output_dir_path: {output_dir_path}")
    print(f"frame_step: {frame_step}")
    print(f"angles: {angles}")

    est_angles = {
        "left_knee_angles": [],
        "right_knee_angles": [],
    }

    video_reader = OpenCV.VideoCapture(video_file_path)
    width = int(video_reader.get(OpenCV.CAP_PROP_FRAME_WIDTH))
    height = int(video_reader.get(OpenCV.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(OpenCV.CAP_PROP_FPS)
    frames = int(video_reader.get(OpenCV.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"- Dimensions: {width}x{height}")
    print(f"- Total frames: {frames}")
    print(f"- FPS: {fps}")
    print(f"- Duration: {frames/fps:.2f} seconds")

    frame_step = max(frames // frame_step, 1)
    print(f"video frame step: {frame_step}")

    if width > 1000 and height > 1000:
        size = (width // 2, height // 2)
    else:
        size = (width, height)
    print(f"video frame size: {size}")

    # Create video writer for landmark video
    landmark_video_path = f"{output_dir_path}/landmark_video.mp4"
    video_writer = OpenCV.VideoWriter(
        landmark_video_path, OpenCV.VideoWriter_fourcc(*"mp4v"), fps, size
    )

    frame_index = 0

    min_angle = float("inf")
    max_angle = float("-inf")
    min_frame_plain = None
    min_frame_landmarks = None
    max_frame_plain = None
    max_frame_landmarks = None
    min_angle_type = None
    max_angle_type = None

    while video_reader.isOpened():
        keep_going, frame = video_reader.read()

        if not keep_going:
            if frame_index != 0:
                print(f"video processed -> {landmark_video_path}")
            else:
                print(f"could not process {video_file_path}")
            break

        frame_index += 1
        if frame_index % frame_step != 0:
            continue

        frame = OpenCV.resize(frame, size)
        original_frame = frame.copy()  # Keep a clean copy for plain frames

        # Process frame for pose detection and angle calculation
        frame_with_landmarks, est_angles = estimate_pose(frame, angles, est_angles)

        if frame_with_landmarks is not None:
            # Write frame to landmark video
            video_writer.write(frame_with_landmarks)

            # Check for min/max angles and store frames
            for angle_type, angle_list in est_angles.items():
                if angle_list and len(angle_list) > 0:
                    current_angle = angle_list[-1]  # Get most recent angle

                    if current_angle < min_angle:
                        min_angle = current_angle
                        min_frame_plain = original_frame.copy()
                        min_frame_landmarks = frame_with_landmarks.copy()
                        min_angle_type = angle_type

                    if current_angle > max_angle:
                        max_angle = current_angle
                        max_frame_plain = original_frame.copy()
                        max_frame_landmarks = frame_with_landmarks.copy()
                        max_angle_type = angle_type

    # Release video resources
    video_reader.release()
    video_writer.release()

    # Save min/max frames with both plain and landmark versions
    if min_frame_plain is not None:
        min_plain_path = f"{output_dir_path}/minPlain.png"
        OpenCV.imwrite(min_plain_path, min_frame_plain)
        print(
            f"Saved min angle plain frame ({min_angle_type}: {min_angle:.1f}°) -> {min_plain_path}"
        )

    if min_frame_landmarks is not None:
        min_landmarks_path = f"{output_dir_path}/minImage.png"
        OpenCV.imwrite(min_landmarks_path, min_frame_landmarks)
        print(f"Saved min angle landmarks frame -> {min_landmarks_path}")

    if max_frame_plain is not None:
        max_plain_path = f"{output_dir_path}/maxPlain.png"
        OpenCV.imwrite(max_plain_path, max_frame_plain)
        print(
            f"Saved max angle plain frame ({max_angle_type}: {max_angle:.1f}°) -> {max_plain_path}"
        )

    if max_frame_landmarks is not None:
        max_landmarks_path = f"{output_dir_path}/maxImage.png"
        OpenCV.imwrite(max_landmarks_path, max_frame_landmarks)
        print(f"Saved max angle landmarks frame -> {max_landmarks_path}")

    print(f"video processed -> {video_file_path}")
    return est_angles


def draw_leg_landmarks(frame, pose_detection, side):
    """
    Draw only the leg landmarks on the frame.

    Args:
        frame: The frame to draw on
        pose_detection: The PoseDetection instance
        side: Either "left" or "right"
    """
    mp_pose = pose_detection.mp_pose
    mp_draw = pose_detection.mp_draw

    # Define connections for legs
    leg_connections = []
    if side == "left":
        leg_connections = [
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
            (mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
            (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
        ]
    elif side == "right":
        leg_connections = [
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
            (mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
            (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
        ]

    # Draw only specified landmarks
    landmarks = pose_detection.pose_landmarks
    if landmarks:
        # Draw landmarks
        for i, landmark in enumerate(landmarks.landmark):
            # Only draw hip, knee, ankle, heel and foot index
            if (side == "left" and i in [23, 25, 27, 29, 31]) or (
                side == "right" and i in [24, 26, 28, 30, 32]
            ):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                OpenCV.circle(frame, (cx, cy), 5, (255, 0, 0), OpenCV.FILLED)

        # Draw connections
        for connection in leg_connections:
            start = connection[0].value
            end = connection[1].value

            if 0 <= start < len(landmarks.landmark) and 0 <= end < len(
                landmarks.landmark
            ):
                h, w, c = frame.shape

                start_point = landmarks.landmark[start]
                end_point = landmarks.landmark[end]

                cx1, cy1 = int(start_point.x * w), int(start_point.y * h)
                cx2, cy2 = int(end_point.x * w), int(end_point.y * h)

                OpenCV.line(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)


def estimate_pose(frame, angles: List[str], est_angles: Dict[str, List[float]]):
    processed_frame = None

    try:
        rgb_frame = OpenCV.cvtColor(frame, OpenCV.COLOR_BGR2RGB)
        pose_detection = pose.PoseDetection()

        # Process frame but don't draw landmarks yet
        detection_result = pose_detection.process_frame(rgb_frame, draw=False)

        if not detection_result.pose_landmarks:
            print("Could not process frame.")
            return None, est_angles

        # Get the processed frame without landmarks
        processed_frame = frame.copy()

        # knee landmarks
        left_knee_angle = ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE")
        right_knee_angle = ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")

        for angle in angles:
            if angle == "left_knee":
                if all(
                    landmark in pose_detection.landmark_positions
                    for landmark in left_knee_angle
                ):
                    pose_detection.draw_angle(left_knee_angle)
                    # Draw only the left leg landmarks
                    draw_leg_landmarks(processed_frame, pose_detection, "left")
                    est_angles["left_knee_angles"].append(
                        float(f"{pose_detection.angle(left_knee_angle):.1f}")
                    )
                else:
                    missing = [
                        lm
                        for lm in left_knee_angle
                        if lm not in pose_detection.landmark_positions
                    ]
                    print(f"Missing landmarks for left knee angle: {missing}")

            elif angle == "right_knee":
                if all(
                    landmark in pose_detection.landmark_positions
                    for landmark in right_knee_angle
                ):
                    pose_detection.draw_angle(right_knee_angle)
                    # Draw only the right leg landmarks
                    draw_leg_landmarks(processed_frame, pose_detection, "right")
                    est_angles["right_knee_angles"].append(
                        float(f"{pose_detection.angle(right_knee_angle):.1f}")
                    )
                else:
                    missing = [
                        lm
                        for lm in right_knee_angle
                        if lm not in pose_detection.landmark_positions
                    ]
                    print(f"Missing landmarks for right knee angle: {missing}")
            else:
                print(f"unknown angle: {angle}")

    except Exception as e:
        print(f"Error occurred in estimate_pose: {e}")
        import traceback

        print(traceback.format_exc())

    return processed_frame if processed_frame is not None else None, est_angles
