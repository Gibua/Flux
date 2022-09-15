from abc import ABC, abstractmethod
from typing import Optional, Tuple

class LandmarkPredictor(ABC):

    @property
    def landmark_count(self):
        pass

    @property
    def dataset(self):
        pass

    @property
    def pose_is_provided(self):
        pass

    @abstractmethod
    def predict(self, img, bbox):
        pass

    @abstractmethod
    def _pre_process_bbox(self, bbox, frame_shape: Optional[Tuple[int, int]] = None):
        pass
        
    @abstractmethod
    def pre_process(self, img, bbox):
        pass
    
    @abstractmethod
    def post_process(self, landmarks, bbox):
        pass
        
    def get_eye_indices(self):
        pass
        
    def get_eye_corner_indices(self):
        pass
