# Object-Tracking and Counting using Deep Sort

This repository implements YOLOv3 and Deep SORT in order to perform real-time object tracking and counting.

![](https://github.com/Akhil-Tony/Object-Detection-Object-Tracking-and-Counting/blob/master/deep_sort/architecture.jpg)

## Installation

First, clone this GitHub repository. Install requirements.

Then download pretrained weights and place it to model_data directory:
```
# yolov3
wget -P model_data https://pjreddie.com/media/files/yolov3.weights
``````

## Testing

```
python object_tracker.py
````

## Result
![](https://github.com/Akhil-Tony/Object-Detection-Object-Tracking-and-Counting/blob/master/track_1.gif) 
![](https://github.com/Akhil-Tony/Object-Detection-Object-Tracking-and-Counting/blob/master/track_3.gif)

## Thoughts
Training the yolo classifier backbone with additional data of cars and trucks will help 
reduce the misclassification between car and truck.
Few false tracking box are predicted
## References
1. Deep SORT Repository - https://github.com/anushkadhiman/ObjectTracking-DeepSORT-YOLOv3-TF2
