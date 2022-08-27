# Object-Detection-Object-Tracking-and-Counting

This repository implements YOLOv3 and Deep SORT in order to perform real-time object tracking.

## Installation

First, clone or download this GitHub repository. Install requirements.

Then download pretrained weights to the model_data directory:
```
# yolov3
wget -P model_data https://pjreddie.com/media/files/yolov3.weights
``````

## Testing

```
python object_tracker.py
````

## Result

### Tracking and Counting
![Alt text](tracking.gif?raw=true "video")

## References
1. Deep SORT Repository - https://github.com/anushkadhiman/ObjectTracking-DeepSORT-YOLOv3-TF2







