#from ..UltraLightFace.utils import *
#from ..UltraLightFace.TFLiteFaceDetector import UltraLightFaceDetecion
#from modules.faster_retinaface.face_detector import MxnetDetectionModel
from modules.faster_retinaface import face_detector
import os
import numpy as np

import time


class Detector:
    
    def __init__(self):    
        self.weight_path = os.path.abspath("./modules/faster_retinaface/weights/16and32")
        self._detector = face_detector.MxnetDetectionModel(self.weight_path, 0, scale=.4, gpu=-1, margin=0.15, nms_thd=0.4)
        
    def detect_bbox(self, img):
        #start_t = time.perf_counter()
        dets = self._detector.detect(img)
        #print(time.perf_counter() - start_t)
        boxes = np.array(list(dets))

        if boxes.size == 0:
            return False, np.empty(0)
        else:
            return True, boxes

        
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
