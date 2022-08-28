# Object-Tracking and Counting using Deep Sort

This repository implements YOLOv3 and Deep SORT in order to perform real-time object tracking and counting.

![](https://www.researchgate.net/publication/353256407/figure/fig2/AS:1045653165715457@1626314550831/Architecture-of-Deep-SORT-Simple-online-and-real-time-tracking-with-deep-association.jpg)

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

## Observations
### Video 1
__Some cars are misclassified as trucks, Training the classifier backbone with more data containing cars and trucks may help.__
### Video 3
__Few false tracking box are predicted__
## References
1. Deep SORT Repository - https://github.com/anushkadhiman/ObjectTracking-DeepSORT-YOLOv3-TF2
