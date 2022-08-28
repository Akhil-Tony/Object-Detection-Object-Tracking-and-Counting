'''
Custom Module
'''
'''utility function to load yolo class names'''
def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

class Count:
    CLASSES = None
    scale = None
    
    def __init__(self, scale):
        self.CLASSES = read_class_names("deep_sort/yolo_classes/coco.names")
        self.scale = scale
        
    '''for each bounding box the function identify unique objects passing the crossing line and returns the count and the uniques objects'''
    def get_count(self,image_shape, bboxes, object_set, object_count):
        frame_height,_,_ = image_shape
        '''uniquely identify objects which crosses the crossing line on the road'''
        y_line = int(frame_height * self.scale)
        
        for bbox in bboxes:
            class_ind = int(bbox[5])
            class_name = self.CLASSES[class_ind]
            id_num = bbox[4]
            y_center = (bbox[1] + bbox[3])/2 # center point for the object
            set_element = class_name+str(id_num) 
            '''checks if the object passes the crossing line'''
            if y_center <= y_line:
                '''if the object is not identified before then it is added to the set and increments the count for its category'''
                if set_element not in object_set: 
                    object_set.add(set_element)
                    object_count[class_name] += 1

        return object_count,object_set
