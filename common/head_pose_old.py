import numpy as np
import eos
import os
import cv2

from common.landmark_mapping import LandmarkMapper
from common.mappings import Datasets
from common.face_model import FaceModel68

class PnPHeadPoseEstimator(object):
    ibug_ids_to_use = sorted([
        28, 29, 30, 31,  # nose ridge
        32, 33, 34, 35, 36,  # nose base
        37, 40,  # left-eye corners
        43, 46   # right-eye corners
    ])

    def __init__(self):
        # Load and extract vertex positions for selected landmarks
        cwd = os.path.dirname(__file__)
        base_dir = cwd
        self.model = eos.morphablemodel.load_model(
            base_dir + '/share/sfm_shape_3448.bin')
        self.shape_model = self.model.get_shape_model()
        self.landmarks_mapper = eos.core.LandmarkMapper(
            base_dir + '/share/ibug_to_sfm.txt')
        self.sfm_points_ibug_subset = np.array([
            self.shape_model.get_mean_at_point(
                int(self.landmarks_mapper.convert(str(d)))
            )
            for d in range(1, 69)
            if self.landmarks_mapper.convert(str(d)) is not None
        ])

        self.sfm_points_for_pnp = np.array([
            self.shape_model.get_mean_at_point(
                int(self.landmarks_mapper.convert(str(d)))
            )
            for d in self.ibug_ids_to_use
        ])

        # Rotate face around
        rotate_mat = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        self.sfm_points_ibug_subset = np.matmul(self.sfm_points_ibug_subset.reshape(-1, 3), rotate_mat)
        self.sfm_points_for_pnp = np.matmul(self.sfm_points_for_pnp.reshape(-1, 3), rotate_mat)

        # Center on mean point between eye corners
        between_eye_point = np.mean(self.sfm_points_for_pnp[-4:, :], axis=0)
        self.sfm_points_ibug_subset -= between_eye_point.reshape(1, 3)
        self.sfm_points_for_pnp -= between_eye_point.reshape(1, 3)

    def fit_func(self, landmarks, camera_matrix):
        landmarks = np.array([
            landmarks[i - 1, :]
            for i in self.ibug_ids_to_use
        ], dtype=np.float64)

        # Initial fit
        success, rvec, tvec, inliers = cv2.solvePnPRansac(self.sfm_points_for_pnp, landmarks,
                                                          camera_matrix, None, flags=cv2.SOLVEPNP_EPNP)

        # Second fit for higher accuracy
        success, rvec, tvec = cv2.solvePnP(self.sfm_points_for_pnp, landmarks, camera_matrix, None,
                                           rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

        return rvec, tvec

    def project_model(self, rvec, tvec, camera_matrix):
        points, _ = cv2.projectPoints(self.sfm_points_ibug_subset, rvec, tvec, camera_matrix, None)
        return points


    def drawPose(self, img, r, t, cam_mat, dist = np.zeros((4, 1))):

        modelAxes = np.array([
            np.array([0., -20., 0.]).reshape(1, 3),
            np.array([50., -20., 0.]).reshape(1, 3),
            np.array([0., -70., 0.]).reshape(1, 3),
            np.array([0., -20., -50.]).reshape(1, 3)
        ])

        projAxes, jac = cv2.projectPoints(modelAxes, r, t, cam_mat, dist)

        cv2.line(img, (int(projAxes[0, 0, 0]), int(projAxes[0, 0, 1])),
                 (int(projAxes[1, 0, 0]), int(projAxes[1, 0, 1])),
                 (0, 255, 255), 2)
        cv2.line(img, (int(projAxes[0, 0, 0]), int(projAxes[0, 0, 1])),
                 (int(projAxes[2, 0, 0]), int(projAxes[2, 0, 1])),
                 (255, 0, 255), 2)
        cv2.line(img, (int(projAxes[0, 0, 0]), int(projAxes[0, 0, 1])),
                 (int(projAxes[3, 0, 0]), int(projAxes[3, 0, 1])),
                 (255, 255, 0), 2)
