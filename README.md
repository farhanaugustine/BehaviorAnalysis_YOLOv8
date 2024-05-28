
![Yolov8_Enteries_Exits_FullWorkflow](https://github.com/farhanaugustine/BehaviorAnalysis_YOLOv8/assets/54376988/15306884-2468-48a5-b88d-5920f759b032)

[![DOI](https://zenodo.org/badge/803993618.svg)](https://zenodo.org/doi/10.5281/zenodo.11264288)


# YoloV8_OFT_MultiROI

## Youtube Video Tutorial:
[![Single Animal Behavior Analysis (Yolov8)](https://i9.ytimg.com/vi_webp/DydI37F5nVM/mq1.webp?sqp=CKS7wLIG-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGH8gEygfMA8=&rs=AOn4CLDH58muj8QNINXxeh37t84fUyQtLA)](https://youtu.be/DydI37F5nVM)

https://youtu.be/DydI37F5nVM

# Object Detection and Tracking with YOLO and OpenCV

This script uses the YOLO (You Only Look Once) model from the Ultralytics library and OpenCV to perform object detection and tracking on a video. The detected objects are then analyzed based on their position in relation to predefined Regions of Interest (ROIs).

## Dependencies

- Ultralytics YOLO
- OpenCV
- NumPy
- csv

## How it works

1. **Load the YOLO model**: The YOLO model is loaded from the specified path. This model will be used to make predictions on the video frames.

2. **Load the video**: The video is loaded from the specified path using `cv2.VideoCapture()`. This function returns a `VideoCapture` object which is used to read the video frames.

3. **Initialize frame counter**: A frame counter is initialized to keep track of the current frame number.

4. **Create a VideoWriter object**: Two `VideoWriter` objects are created using `cv2.VideoWriter()`. These objects are used to save the mask video and the video with ROIs and counts drawn on top.

5. **Define multiple ROIs**: Multiple ROIs are defined as dictionaries. Each ROI has a top left and bottom right coordinate, as well as counters for entries and exits, a flag for whether the previous position was inside the ROI, frame counters for entries and exits, a counter for total frames, a counter for total distance, and a variable for the previous position.

6. **Create a CSV file and write the header**: A CSV file is created to store the frame number and the center coordinates of the detected objects.

7. **Read frames**: The script enters a loop where it reads frames from the video one by one.

8. **Object detection**: The YOLO model is used to make predictions on the current frame. If any objects are detected, their masks are processed and their centers are calculated.

9. **ROI analysis**: For each detected object, the script checks if its center is inside any of the ROIs. If the object has entered or exited an ROI, the corresponding counters are updated. The total distance traveled by the object inside the ROI is also calculated.

10. **Draw on frame**: The ROIs and the centers of the detected objects are drawn on the frame. The entry and exit counts for each ROI are also drawn on the frame.

11. **Save frame and mask to video**: The frame with the drawn ROIs and counts, as well as the mask of the detected objects, are saved to video files using the `VideoWriter` objects.

12. **Increment frame counter**: The frame counter is incremented.

13. **Write CSV data to file**: After all frames have been processed, the CSV data is written to the file.

14. **Release resources**: The `VideoCapture` and `VideoWriter` objects are released and any openCV windows are destroyed.

15. **Print ROI counts**: The entry and exit counts, total frames spent, and average velocity for each ROI are printed.

16. **Write ROI counts to CSV file**: The entry and exit counts, total frames spent, and average velocity for each ROI are written to the CSV file.

## Usage

To run the script, simply execute it with Python:

```bash
python yolov8_seg.py
