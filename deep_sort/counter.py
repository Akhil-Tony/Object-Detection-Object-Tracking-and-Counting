
def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

YOLO_COCO_CLASSES = read_class_names("deep_sort/yolo_classes/coco.names")

def get_count(image, bboxes, object_set, object_count, scale, CLASSES=YOLO_COCO_CLASSES):
    frame_height,_,_ = image.shape
    y_line = int(frame_height * scale)
    for bbox in bboxes:
        class_ind = int(bbox[5])
        class_name = CLASSES[class_ind]
        id_num = bbox[4]
        y_center = (bbox[1] + bbox[3])/2
        set_element = class_name+str(id_num)
        if y_center <= y_line:
            if set_element not in object_set:
                object_set.add(set_element)
                object_count[class_name] += 1
    
    return object_count,object_set

        



