import numpy as np

class PinholeCamera:

    def __init__(self, width: int, height: int, camera_matrix = None):
        if camera_matrix is None:
            self.camera_matrix = self.estimate_camera_matrix(width, height)
        else:
            self.camera_matrix = camera_matrix
        self.width = width
        self.height = height
        self.dist_coefficients = np.zeros(5, dtype=np.float)
  

    @staticmethod
    def estimate_camera_matrix(width: int, height: int):
        c_x = width / 2
        c_y = height / 2
        FieldOfView = 60
        focal = c_x / np.tan(np.radians(FieldOfView/2))
        camera_matrix = np.float32([[focal, 0.0,   c_x], 
                                    [0.0,   focal, c_y],
                                    [0.0,   0.0,   1.0]])
        return camera_matrix

    def get_focal(self):
        return self.camera_matrix[0][0]
