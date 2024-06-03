# YOLO Object Detection

## Project Overview
This project demonstrates an object detection system using a pre-trained YOLOv8 model. It processes video frames, detects objects, and displays bounding boxes with class labels and confidence scores. The detected objects are highlighted with randomly generated colors for better visualization.

## Key Features
- Uses a pre-trained YOLOv8 model for object detection.
- Reads and processes video frames from a file.
- Displays detected objects with bounding boxes and class labels.
- Generates random colors for different classes for better visualization.
- Supports real-time detection with video playback.

## Installation
To get started with this project, follow the steps below:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/yolo-object-detection.git
    cd yolo-object-detection
    ```

2. **Install the required libraries**:
    Ensure you have Python installed, then run:
    ```sh
    pip install ultralytics opencv-python-headless
    ```

3. **Download the YOLOv8 model**:
    Make sure you have the `best.pt` YOLOv8 model file in your project directory.

4. **Prepare the COCO class list**:
    Ensure the `utils/coco.txt` file contains the COCO class names, one per line.

## Usage
To run the object detection script on a video file, follow these steps:

1. **Place your video file**:
    Ensure you have a video file named `he2.mp4` in the project directory. You can replace it with any other video file and update the script accordingly.

2. **Run the script**:
    Execute the Python script to start object detection:
    ```sh
    python detect.py
    ```

3. **Interact with the output**:
    The script will display the video with detected objects highlighted. Press `Q` to terminate the video playback.

## Script Details
The script performs the following operations:

1. **Initialization**:
    - Reads the COCO class names from `utils/coco.txt`.
    - Generates random colors for each class.
    - Loads the pre-trained YOLOv8 model.

2. **Video Processing**:
    - Captures frames from the video file.
    - Resizes frames for optimized processing.
    - Predicts objects in each frame using the YOLOv8 model.
    - Draws bounding boxes and labels on detected objects.
    - Displays the processed frame in a window.

3. **Termination**:
    - Releases video capture and destroys all windows upon termination.



## Challenges Faced
Some of the challenges encountered during the project include:
- Understanding the YOLOv8 model's output format and processing it correctly.
- Optimizing the video frame processing for real-time performance.
- Handling various edge cases such as video file not found or empty frames.

## Conclusion
This project provided a hands-on experience with YOLO-based object detection and real-time video processing. It showcases the potential of deep learning models in practical applications like object detection and tracking.

## Future Enhancements
- Improve the detection accuracy by fine-tuning the model.
- Add support for real-time detection from a webcam.
- Implement additional features such as object tracking and counting.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [YOLO (You Only Look Once)](https://pjreddie.com/darknet/yolo/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8)
- [OpenCV](https://opencv.org/)

