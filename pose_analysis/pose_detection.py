from typing import Dict, Tuple  # Python 3.8 does not support TypedDict by default

import cv2 as cv
import mediapipe as mp
import numpy as np


class PoseDetection:
    def __init__(self):
        self.landmark_positions: Dict[str, np.ndarray] = {}
        self.pose_landmarks = None
        self.processed_frame = None

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        # Add configuration parameters for better performance
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,  # Use 1 for better performance, 2 for better accuracy
        )

        # https://google.github.io/mediapipe/solutions/pose#pose-landmark-model-blazepose-ghum-3d
        self.landmark_names = {
            self.mp_pose.PoseLandmark.NOSE: "NOSE",
            self.mp_pose.PoseLandmark.LEFT_EYE_INNER: "LEFT_EYE_INNER",
            self.mp_pose.PoseLandmark.LEFT_EYE: "LEFT_EYE",
            self.mp_pose.PoseLandmark.LEFT_EYE_OUTER: "LEFT_EYE_OUTER",
            self.mp_pose.PoseLandmark.RIGHT_EYE_INNER: "RIGHT_EYE_INNER",
            self.mp_pose.PoseLandmark.RIGHT_EYE: "RIGHT_EYE",
            self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER: "RIGHT_EYE_OUTER",
            self.mp_pose.PoseLandmark.LEFT_EAR: "LEFT_EAR",
            self.mp_pose.PoseLandmark.RIGHT_EAR: "RIGHT_EAR",
            self.mp_pose.PoseLandmark.MOUTH_LEFT: "MOUTH_LEFT",
            self.mp_pose.PoseLandmark.MOUTH_RIGHT: "MOUTH_RIGHT",
            self.mp_pose.PoseLandmark.LEFT_SHOULDER: "LEFT_SHOULDER",
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER: "RIGHT_SHOULDER",
            self.mp_pose.PoseLandmark.LEFT_ELBOW: "LEFT_ELBOW",
            self.mp_pose.PoseLandmark.RIGHT_ELBOW: "RIGHT_ELBOW",
            self.mp_pose.PoseLandmark.LEFT_WRIST: "LEFT_WRIST",
            self.mp_pose.PoseLandmark.RIGHT_WRIST: "RIGHT_WRIST",
            self.mp_pose.PoseLandmark.LEFT_PINKY: "LEFT_PINKY",
            self.mp_pose.PoseLandmark.RIGHT_PINKY: "RIGHT_PINKY",
            self.mp_pose.PoseLandmark.LEFT_INDEX: "LEFT_INDEX",
            self.mp_pose.PoseLandmark.RIGHT_INDEX: "RIGHT_INDEX",
            self.mp_pose.PoseLandmark.LEFT_THUMB: "LEFT_THUMB",
            self.mp_pose.PoseLandmark.RIGHT_THUMB: "RIGHT_THUMB",
            self.mp_pose.PoseLandmark.LEFT_HIP: "LEFT_HIP",
            self.mp_pose.PoseLandmark.RIGHT_HIP: "RIGHT_HIP",
            self.mp_pose.PoseLandmark.LEFT_KNEE: "LEFT_KNEE",
            self.mp_pose.PoseLandmark.RIGHT_KNEE: "RIGHT_KNEE",
            self.mp_pose.PoseLandmark.LEFT_ANKLE: "LEFT_ANKLE",
            self.mp_pose.PoseLandmark.RIGHT_ANKLE: "RIGHT_ANKLE",
            self.mp_pose.PoseLandmark.LEFT_HEEL: "LEFT_HEEL",
            self.mp_pose.PoseLandmark.RIGHT_HEEL: "RIGHT_HEEL",
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX: "LEFT_FOOT_INDEX",
            self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX: "RIGHT_FOOT_INDEX",
        }

    def process_frame(self, frame, draw=True):
        """
        Process a frame and extract pose landmarks.

        Args:
                frame (numpy.ndarray): The input frame to process.
                draw (bool, optional): Whether to draw the pose landmarks on the frame. Defaults to True.

        Returns:
                self: The instance of the class with updated pose landmarks and processed frame.
        """
        h, w, c = frame.shape  # Get image dimensions for proper landmark projection

        # Process the frame to detect poses
        results = self.pose.process(frame)
        self.pose_landmarks = results.pose_landmarks

        if not self.pose_landmarks:
            print("Could not process frame.")
            return self

        if draw:
            self.mp_draw.draw_landmarks(
                frame, self.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

        self.processed_frame = frame

        head_top = None
        head_bottom = None
        head_left = None
        head_right = None

        for i, landmark in enumerate(self.pose_landmarks.landmark):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            self.landmark_positions[self.landmark_names[i]] = np.array([cx, cy])

            if self.landmark_names[i] in [
                "LEFT_EAR",
                "RIGHT_EAR",
                "NOSE",
                "LEFT_SHOULDER",
                "RIGHT_SHOULDER",
            ]:
                if head_top is None or cy < head_top:
                    head_top = cy
                if head_bottom is None or cy > head_bottom:
                    head_bottom = cy
                if head_left is None or cx < head_left:
                    head_left = cx
                if head_right is None or cx > head_right:
                    head_right = cx

            if draw:
                cv.circle(self.processed_frame, (cx, cy), 5, (255, 0, 0), cv.FILLED)

        if (
            head_top is not None
            and head_bottom is not None
            and head_left is not None
            and head_right is not None
        ):
            # There is no top of head landmark, so we have to adjust the box up a bit
            height = head_bottom - head_top
            head_top -= height

            head = cv.getRectSubPix(
                self.processed_frame,
                (head_right - head_left, head_bottom - head_top),
                ((head_right + head_left) / 2, (head_bottom + head_top) / 2),
            )

            # Cover the head with a black rectangle
            cv.rectangle(
                self.processed_frame,
                (head_left, head_top),
                (head_right, head_bottom),
                (0, 0, 0),
                -1,
            )

        return self

    def angle(self, landmark_names):
        """
        Calculate the angle between three landmarks.

        Parameters:
        - landmark_names (list): A list of three landmark names.

        Returns:
        - angle (float): The angle between the three landmarks in degrees.
        """
        # Check if all landmarks exist
        if not all(landmark in self.landmark_positions for landmark in landmark_names):
            raise ValueError(f"Not all landmarks found: {landmark_names}")

        a = self.landmark_positions[landmark_names[0]]
        b = self.landmark_positions[landmark_names[1]]
        c = self.landmark_positions[landmark_names[2]]

        ba = a - b
        bc = c - b

        # Prevent division by zero and ensure valid cosine values
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba == 0 or norm_bc == 0:
            return 0.0

        cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
        _angle = np.degrees(np.arccos(cosine_angle))
        _angle = 180 - _angle

        return _angle

    def draw_angle(self, landmark_names: Tuple[str, str, str]):
        """
        Draws an angle value on the processed frame.

        Parameters:
        - landmark_names (Tuple[str, str, str]): A tuple of three landmark names.

        Returns:
        None
        """
        if landmark_names[1] in self.landmark_positions:
            x, y = self.landmark_positions[landmark_names[1]]
            
            # Format the angle label
            try:
                angle_value = self.angle(landmark_names)
                
                # Create more readable label
                joint_name = landmark_names[1].replace('_', ' ').title()
                
                # Draw background for better visibility
                cv.rectangle(
                    self.processed_frame, 
                    (x - 5, y - 35), 
                    (x + 95, y - 5), 
                    (0, 0, 0), 
                    -1
                )
                
                # Draw angle value with joint name
                cv.putText(
                    self.processed_frame,
                    f"{joint_name}",
                    (x, y - 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )
                
                cv.putText(
                    self.processed_frame,
                    f"{angle_value:.1f}",
                    (x, y - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),  # Yellow color for better visibility
                    1,
                    cv.LINE_AA,
                )
                
            except Exception as e:
                print(f"Error calculating angle: {e}")


if __name__ == "__main__":
    pass
