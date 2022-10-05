import os
import time
import numpy as np
import pickle
import cv2
from scipy.spatial.transform import Rotation as R

from common.camera import PinholeCamera
from common.face_model_68 import FaceModel68


def draw_axes(image, rvec: np.ndarray, center: np.ndarray, scale = 100):
    axes_points = np.array([[0., 0., 0.],
                            [1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.] ])
    axes_points *= scale
    rmat = cv2.Rodrigues(rvec)[0]
    print(np.concatenate([center, [0]]))
    proj = ((axes_points.dot(rmat.T) + np.concatenate([center, [0]]))).astype(int)

    cv2.line(image, (proj[0][0], proj[0][1]), (proj[3][0],proj[3][1]),(255,0,0),4)
    cv2.line(image, (proj[0][0], proj[0][1]), (proj[2][0],proj[2][1]),(0,255,0),4)
    cv2.line(image, (proj[0][0], proj[0][1]), (proj[1][0],proj[1][1]),(0,0,255),4)


def draw_axes_cv(image, rotation_vector, translation_vector, camera: PinholeCamera):
    cam_mat = camera.camera_matrix
    dist_coeffs = camera.dist_coefficients

    cv2.drawFrameAxes(image, cam_mat, dist_coeffs, rotation_vector, translation_vector, 30)


def draw_angles_text(image, rotation_vector):
    xyz = R.from_rotvec(rotation_vector.flatten()).as_euler('xyz', degrees=True)
    pitch_color = (210,200,0)
    yaw_color   = (50,150,0)
    roll_color  = (0,0,255)
    
    cv2.putText(image, "Pitch:{:.2f}".format(xyz[0]), (0,10), cv2.FONT_HERSHEY_PLAIN, 1, pitch_color)
    cv2.putText(image, "Yaw:{:.2f}".format(xyz[1]), (0,25), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
    cv2.putText(image, "Roll:{:.2f}".format(xyz[2]), (0,40), cv2.FONT_HERSHEY_PLAIN, 1, roll_color)


def draw_annotation_box(image, rotation_vector, translation_vector, camera: PinholeCamera, color=(255, 255, 255), line_width=2):
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
                                 camera.camera_matrix,
                                 camera.dist_coefficients)[0]
    point_2d = point_2d.reshape(-1, 2).astype(int)
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


class ScaledOrthoParameters:
    """
    Parameters of an estimated scaled orthographic projection.
    
    Attributes:
        rvec: Rotation vector
        tx: Translation in x
        ty: Translation in y
        s: Scaling
    """
    def __init__(self, r=np.zeros((3, 1)), tx=0.0, ty=0.0, s=0.0):
        self.R = r
        self.tx = tx
        self.ty = ty
        self.s = s


def estimate_orthographic_projection_linear(image_points, model_points, is_viewport_upsidedown, viewport_height=None):
    """
    Sourced from: https://github.com/Yinghao-Li/3DMM-fitting/blob/master/core/orthographic_camera_estimation_linear.py
    Which in turn was adapted from: https://github.com/patrikhuber/eos/blob/master/include/eos/fitting/orthographic_camera_estimation_linear.hpp

    Estimates the parameters of a scaled orthographic projection.
    
    Given a set of 2D-3D correspondences, this algorithm estimates rotation,
    translation (in x and y) and a scaling parameters of the scaled orthographic
    projection model using a closed-form solution. It does so by first computing
    an affine camera matrix using algorithm [1], and then finds the closest
    orthonormal matrix to the estimated affine transform using SVD.
    This algorithm follows the original implementation [2] of William Smith,
    University of York.
    
    Requires >= 4 corresponding points.
    
    [1]: Gold Standard Algorithm for estimating an affine camera matrix from
    world to image correspondences, Algorithm 7.2 in Multiple View Geometry,
    Hartley & Zisserman, 2nd Edition, 2003.
    [2]: https://github.com/waps101/3DMM_edges/blob/master/utils/POS.m
    
    Args:
        image_points:
            A list of 2D image points, with the shape of nx2.
        model_points:
            Corresponding points of a 3D model, with the shape of nx4 (homogeneous).
        is_viewport_upsidedown:
            Flag to set whether the viewport of the image points is upside-down (e.g. as in OpenCV).
        viewport_height:
            Height of the viewport of the image points (needs to be given if is_viewport_upsidedown == true).
        
    Returns:
        Rotation, translation and scaling of the estimated scaled orthographic projection.
    """
    assert len(image_points) == len(model_points)
    # Number of correspondence points given needs to be equal to or larger than 4
    assert len(image_points) >= 4
    
    num_correspondences = len(image_points)
    
    image_points = np.array(image_points)
    model_points = np.array(model_points)
    
    # TODO: Might be problematic, should be noticed!
    if is_viewport_upsidedown:
        if viewport_height is None:
            raise RuntimeError('Error: If is_viewport_upsidedown is set to true, viewport_height needs to be given.')
        for ip in image_points:
            ip[1] = viewport_height - ip[1]
    
    # Build linear system of equations in 8 unknowns of projection matrix
    a = np.ones([2 * num_correspondences, 8])
    # !! This part was wrong, and has been corrected.
    homogeneous = np.append(model_points.copy(), np.zeros((model_points.shape[0],1), dtype=float), axis=1)
    print(homogeneous)
    a[0: 2 * num_correspondences: 2, :4] = homogeneous
    a[1: 2 * num_correspondences: 2, 4:] = homogeneous
    
    # TODO: Is it necessary?
    b = np.reshape(image_points, [2 * num_correspondences])
    
    # Using pseudo-inverse matrix (sdv) to solve linear system
    k = np.linalg.lstsq(a, b)[0]
    
    # Extract params from recovered vector
    r_1 = k[0:3]
    r_2 = k[4:7]
    stx = k[3]
    sty = k[7]
    s = (np.linalg.norm(r_1) + np.linalg.norm(r_2)) / 2
    r1 = r_1 / np.linalg.norm(r_1)
    r2 = r_2 / np.linalg.norm(r_2)
    r3 = np.cross(r1, r2)
    r = np.array([r1, r2, r3])
    
    # Set R_ortho to closest orthogonal matrix to estimated rotation matrix
    [u, _, vt] = np.linalg.svd(r)
    r_ortho = u.dot(vt)
    
    # The determinant of r must be 1 for it to be a valid rotation matrix
    if np.linalg.det(r_ortho) < 0:
        u[2, :] = -u[2, :]
        r_ortho = u.dot(vt)
    
    # Remove the scale from the translations
    tx = stx / s
    ty = sty / s
    
    return cv2.Rodrigues(r_ortho)[0], tx, ty, s


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

    def __init__(self,  camera: PinholeCamera):
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
        #self.model_points_68 = FaceModel68.LANDMARKS.copy()*1000 #the model is in millimeters
        print('open is assigned to %r' % open)
        with open("./common/ICTFaceModel.pkl", "rb") as ICT_file:
            ICT_Model = pickle.load(ICT_file)
            ICT_to_iBUG_idxs = [1225,1888,1052,367,1719,1722,2199,1447,966,3661,4390,3927,3924,2608,3272,4088,3443,268,493,1914,2044,1401,3615,4240,4114,2734,2509,978,4527,4942,4857,1140,2075,1147,4269,3360,1507,1542,1537,1528,1518,1511,3742,3751,3756,3721,3725,3732,5708,5695,2081,0,4275,6200,6213,6346,6461,5518,5957,5841,5702,5711,5533,6216,6207,6470,5517,5966]
            self.model_points_68 = ICT_Model['neutral'][ICT_to_iBUG_idxs]

        self.calibration_rvec = np.zeros((3, 1))


    def set_img_shape(self, width: int, height: int):
        self.camera = PinholeCamera(width, height)


    def set_calibration(self, landmarks_2D: np.ndarray):
        self.calibration_rvec = self.solve_pose(landmarks_2D, calibration = False)[0]

    #def set_calibration(self, rvec: np.ndarray):
    #    self.calibration_rvec = rvec.copy()


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


    def project_model(self, rvec, tvec):
        cam_mat = self.camera.camera_matrix
        dist_coeffs = self.camera.dist_coefficients
        (point_2d, _) = cv2.projectPoints(self.model_points_68,
                                          rvec, tvec,
                                          cam_mat,
                                          dist_coeffs)
        point_2d = point_2d.reshape(-1, 2)

        return point_2d


class EOSEstimator():
    def __init__(self,  camera: PinholeCamera):
        # iBUG indexes for PNP estimation.
        #self.pnp_indexes = np.array([27, 28, 29, 30,  # nose ridge
        #                             31, 33, 35,      # nose base right, center and left
        #                             36, 39,          # right eye corners
        #                             42, 45]          # left eye corners
        #                             ).astype(int)
        eos_share_dir = os.path.realpath("./modules/eos/share")

        
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