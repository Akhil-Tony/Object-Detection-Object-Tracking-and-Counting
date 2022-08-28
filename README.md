# Object-Detection-Object-Tracking-and-Counting

This repository implements YOLOv3 and Deep SORT in order to perform real-time object tracking and counting.

## Installation

First, clone this GitHub repository. Install requirements.

Then download pretrained weights and place is to model_data directory:
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

## Observations 
### Video 1
- cars are misclassified as trucks
### Video 3
- false tracking bounding box for objects
## References
1. Deep SORT Repository - https://github.com/anushkadhiman/ObjectTracking-DeepSORT-YOLOv3-TF2
