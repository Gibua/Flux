import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from common.camera import Camera
from common.face_model_68 import FaceModel68

import time

class PoseEstimator2D:
    """
        Adapted from: https://github.com/AmrElsersy/PFLD-Pytorch-Landmarks/blob/master/euler_angles.py (by Amr Elsersy; email: amrelsersay@gmail.com)

        -----------------------------------------------------------------------------------
        Description: Head Pose Euler Angles(pitch yaw roll) estimation from 2D-3D correspondences landmarks

        Head Pose Estimation from landmarks annotations with solvePnP OpenCV
        Pitch, Yaw, Roll Rotation angles (euler angles) from 2D-3D Correspondences Landmarks

        Givin a General 3D Face Model (3D Landmarks) & annotations 2D landmarks
        2D point = internsic * exterinsic * 3D point_in_world_space
        if we have 2D - 3D correspondences & internsic camera matrix,
        we can use cv2.solvPnP to get the extrinsic matrix that convert the world_space to camera_3D_space
        this extrinsic matrix is considered as the 3 euler angles & translation vector

        we can do that because the faces have similar 3D structure and emotions & iluminations dosn't affect the pose
        Notes:
            we can choose get any 3D coord from any 3D face model .. changing translation will not affect the angle
            it will only inroduce a bigger tvec but the same rotation matrix
    """

    def __init__(self,  camera: Camera):
        # iBUG indexes for PNP estimation.
        #self.pnp_indexes = np.array([27, 28, 29, 30,  # nose ridge
        #                             31, 33, 35,      # nose base right, center and left
        #                             36, 39,          # right eye corners
        #                             42, 45]          # left eye corners
        #                             ).astype(int)

        self.pnp_indexes = np.array([8,               # chin
                                     27, 28, 29, 30,  # nose ridge
                                     31, 33, 35,      # nose
                                     36, 39,          # right eye corners
                                     42, 45]          # left eye corners
                                     ).astype(int)

        self.camera = camera

        # 3D Face model 3D landmarks
        self.model_points_68 = FaceModel68.LANDMARKS.copy()*1000 #the model is in millimeters

        self.calibration_rvec = np.zeros((3, 1))


    def set_img_shape(self, width: int, height: int):
        self.camera = Camera(width, height)


    def set_calibration(self, landmarks_2D: np.ndarray):
        self.calibration_rvec = self.solve_pose(landmarks_2D, calibration = False)[0]


    def solve_pose(self, landmarks: np.ndarray, calibration: bool = False):
        """
            Estimates head rotation by fitting a 3D model to 2D points

            ----------
            landmarks_2D: numpy array of shape(68, 2) conforming the Multi-PIE/iBUG set of points
            calibration: enables calibration (set calibration values with set_calibration())

            Returns:
            -------
            rvec: rotation numpy array that transforms model space to camera space (3D in both)
            tvec: translation numpy array that transforms model space to camera space
        """

        landmarks_3D = self.model_points_68[self.pnp_indexes].copy()
        landmarks_2D = landmarks[self.pnp_indexes].copy()
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(landmarks_3D, landmarks_2D,
                                                          self.camera.camera_matrix, None,
                                                          flags=cv2.SOLVEPNP_SQPNP)
        
        #success, detected_rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D,
        #                                            self.camera.camera_matrix, None,
        #                                            rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
        #                                            flags=cv2.SOLVEPNP_ITERATIVE)
        detected_rvec, tvec = cv2.solvePnPRefineVVS(landmarks_3D, landmarks_2D,
                                       self.camera.camera_matrix, None,
                                       rvec, tvec)
        if calibration:
            rvec = detected_rvec - self.calibration_rvec
        else:
            rvec = detected_rvec

        #pitch_yaw_roll = R.from_rotvec(rvec.flatten()).as_euler('xyz', degrees=True)

        return rvec, tvec


    def solve_pose_68_points(self, landmarks, calibration = False):

        landmarks_3D = self.model_points_68.copy()
        landmarks_2D = landmarks.copy()

        _, detected_rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, self.camera.camera_matrix, distCoeffs=None)
        
        if calibration:
            rvec = detected_rvec - self.calibration_rvec
        else:
            rvec = detected_rvec

        return rvec, tvec


    def draw_axes(self, image, rotation_vector, translation_vector):
        cam_mat = self.camera.camera_matrix
        dist_coeffs = self.camera.dist_coefficients

        cv2.drawFrameAxes(image, cam_mat, dist_coeffs, rotation_vector, translation_vector, 30)


    def draw_angles_text(self, image, rotation_vector):
        xyz = R.from_rotvec(rotation_vector.flatten()).as_euler('xyz', degrees=True)
        pitch_color = (210,200,0)
        yaw_color   = (50,150,0)
        roll_color  = (0,0,255)
    
        cv2.putText(image, "Pitch:{:.2f}".format(xyz[0]), (0,10), cv2.FONT_HERSHEY_PLAIN, 1, pitch_color)
        cv2.putText(image, "Yaw:{:.2f}".format(xyz[1]), (0,25), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
        cv2.putText(image, "Roll:{:.2f}".format(xyz[2]), (0,40), cv2.FONT_HERSHEY_PLAIN, 1, roll_color)


    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 100
        rear_depth = 100
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 0
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

        # Map to 2d image points
        point_2d = cv2.projectPoints(point_3d,
                                     rotation_vector,
                                     translation_vector,
                                     self.camera.camera_matrix,
                                     self.camera.dist_coefficients)[0]
        point_2d = np.int32(point_2d.reshape(-1, 2))
        #point_3d = 10*(R.from_rotvec(rotation_vector.reshape((3,))).apply(point_3d)+15)
        #point_2d = np.delete(point_3d, 2, axis=1).astype(np.int32)
        #print(point_2d+50)
        #point_2d = np.array([point[0] for point in point_2d]).astype(int)
        # Draw all the lines
        cv2.polylines(image, [point_2d], True, (255,0,0), line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)


    def project_model(self, rvec, tvec):
        cam_mat = self.camera.camera_matrix
        dist_coeffs = self.camera.dist_coefficients
        (point_2d, _) = cv2.projectPoints(self.model_points_68,
                                          rvec, tvec,
                                          cam_mat,
                                          dist_coeffs)
        point_2d = point_2d.reshape(-1, 2)

        return point_2d
