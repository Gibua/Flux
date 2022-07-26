#from ..UltraLightFace.utils import *
#from ..UltraLightFace.TFLiteFaceDetector import UltraLightFaceDetecion
from modules.UltraLightFace.utils import *
from modules.UltraLightFace.TFLiteFaceDetector import UltraLightFaceDetecion
import os
import numpy as np


class Detector:
    
    def __init__(self):    
        relative_path = "./modules/UltraLightFace/pretrained/version-RFB-320_without_postprocessing.tflite"
        self.predictor = UltraLightFaceDetecion(os.path.abspath(relative_path), conf_threshold=0.6)
        
    def detect_bbox(self, img):
        boxes, scores = self.predictor.inference(img)
        
        is_face_detected = boxes.size > 0
        
        if is_face_detected:
            #boxes = np.sort(boxes, axis=0)
            #result = boxes[0]
            result = boxes
        else:
            result = None
            
        return is_face_detected, result
        
    def post_process(self, bbox):
        bbox[1] *= (1+0.1)
        bbox[3] *= (1+0.1)
        bbox = self.__square_box(bbox)
        bbox = bbox.astype(int)
        
        return bbox
    
        
    def __square_box(self, result):
        ymin = result[1]
        ymax = result[3]
    
        xmin = result[0]
        xmax = result[2]
    
        height = ymax-ymin
        width = xmax-xmin
        
        if (height>width):
            center = xmin+((xmax-xmin)/2)
            xmin = int(round(center-height/2))
            xmax = int(round(center+height/2))
        else:
            center = ymin+((ymax-ymin)/2)
            ymin = int(round(center-width/2))
            ymax = int(round(center+width/2))    
    
        return np.array([xmin, ymin, xmax, ymax]).astype(int)
