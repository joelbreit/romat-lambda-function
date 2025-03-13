# Tasks

- [ ] Blackout faces
- [ ] Change "framestep" to "framerate" and update the logic to analyze <framerate> number of frames per second for the entire video
- [ ] Add more logging
- [ ] Get rid of warnings
  - [ ] "Failed to set name for thread: mediapipe..."
  - [ ] "Disabling support for feedback tensors"
- [ ] Lambda function is sometimes using the max amount of memory (10240 MB!) e.g. with just 4 frames per second at 44 seconds
  - Currently relying on limiting the number of total frames to 100, but that will not work for longer videos
- [ ] Lambda function takes nearly a minute to execute

## 0.0.3

- [x] Log Lambda cost at the end of the function
- [x] Fix "??" on annotations

## 0.0.2

- [x] Label minImage.png, maxImage.png, and landmark_video.mp4 with joint angles

## 0.0.1

- [x] Rename minLandmarks.png to minImage.png and maxLandmarks.png to maxImage.png