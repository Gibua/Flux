"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Head Pose Euler Angles(pitch yaw roll) estimation from 2D-3D correspondences landmarks
"""

import cv2
import numpy as np

class EulerAngles:
    """
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

    def __init__(self,  img_shape=(112,112) ):
        # Lazy Estimation of Camera internsic Matrix Approximation
        self.camera_intrinsic_matrix = self.estimate_camera_matrix(img_shape)        
        # 3D Face model 3D landmarks
        self.landmarks_3D = self.get_face_model_3D_landmarks()

    def estimate_camera_matrix(self, img_shape):
        # Used Weak Prespective projection as we assume near object with similar depths

        # cx, cy the optical centres
        # translation to image center as image center here is top left corner
        # focal length is function of image size & Field of View (assumed to be 30 degree)
        c_x = img_shape[0] / 2
        c_y = img_shape[1] / 2
        FieldOfView = 60
        focal = c_x / np.tan(np.radians(FieldOfView/2))
        
        # Approximated Camera intrinsic matrix assuming weak prespective
        return np.float32([
            [focal, 0.0,    c_x], 
            [0.0,   focal,  c_y],
            [0.0,   0.0,    1.0]
        ])

    def set_img_shape(self, img_shape):
        self.camera_intrinsic_matrix = self.estimate_camera_matrix(img_shape)

    def get_face_model_3D_landmarks(self):
        """
            General 3D Face Model Coordinates (3D Landmarks) 
            obtained from antrophometric measurement of the human head.

            Returns:
            -------
            3D_Landmarks: numpy array of shape(N, 3) as N = 11 point in 3D
        """
        # X-Y-Z with X pointing forward and Y on the left and Z up (same as LIDAR)
        # OpenCV Coord X points to the right, Y down, Z to the front (same as 3D Camera)
        
        landmarks_3D = np.float32([
                                   #[-7.308957,  0.913869, 0.000000], #  0 JAWLINE_RIGHT
                                   #[ 7.308957,  0.913869, 0.000000], #  1 JAWLINE_LEFT
                                   [ 6.825897,  6.760612, 4.402142], #  2 LEFT_EYEBROW_LEFT,
                                   [ 1.330353,  7.122144, 6.903745], #  3 LEFT_EYEBROW_RIGHT, 
                                   [-1.330353,  7.122144, 6.903745], #  4 RIGHT_EYEBROW_LEFT,
                                   [-6.825897,  6.760612, 4.402142], #  5 RIGHT_EYEBROW_RIGHT,
                                   [ 5.311432,  5.485328, 3.987654], #  6 LEFT_EYE_LEFT,
                                   [ 1.789930,  5.393625, 4.413414], #  7 LEFT_EYE_RIGHT,
                                   [-1.789930,  5.393625, 4.413414], #  8 RIGHT_EYE_LEFT,
                                   [-5.311432,  5.485328, 3.987654], #  9 RIGHT_EYE_RIGHT,
                                   [ 1.930245,  0.424351, 5.914376], # 10 NOSE_LEFT,
                                   [ 0.000000,  0.000000, 6.763430], # 11 NOSE_CENTER,
                                   [-1.930245,  0.424351, 5.914376], # 12 NOSE_RIGHT,
                                   [ 0.000000,  1.916389, 7.700000], # 13 NOSE_TIP,
                                   #[ 2.774015, -2.080775, 5.048531], # 14 MOUTH_LEFT,
                                   #[-2.774015, -2.080775, 5.048531], # 15 MOUTH_RIGHT,
                                   #[ 0.000000, -1.646444, 6.704956], # 16 UPPER_LIP_CENTER,
                                   #[ 0.000000, -7.415691, 4.070434], # CHIN
                                  ])

        return landmarks_3D


    def euler_angles_from_landmarks(self, landmarks_2D):
        """
            Estimates Euler angles from 2D landmarks 
            
            Parameters:
            ----------
            landmarks_2D: numpy array of shape(N, 2) as N is num of landmarks (usualy 98 from WFLW)

            Returns:
            -------
            rvec: rotation numpy array that transform model space to camera space (3D in both)
            tvec: translation numpy array that transform model space to camera space
            euler_angles: (pitch yaw roll) in degrees
        """

        # WFLW(98 landmark) tracked points
        
        TRACKED_POINTS_MASK = [33, 38, 50, 46, 60, 64, 68, 72, 55, 57, 59, 54]
        #TRACKED_POINTS_MASK = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        #TRACKED_POINTS_MASK = [0, 8, 16, 24, 32, 33, 46, 51, 54, 57, 60, 64, 68, 72, 90]
        landmarks_2D = landmarks_2D[TRACKED_POINTS_MASK]

        """
            solve for extrinsic matrix (rotation & translation) with 2D-3D correspondences
            returns:
                rvec: rotation vector (as rotation is 3 degree of freedom, it is represented as 3d-vector)
                tvec: translate vector (world origin position relative to the camera 3d coord system)
                _ : error -not important-.
        """
        _, rvec, tvec = cv2.solvePnP(self.landmarks_3D, landmarks_2D, self.camera_intrinsic_matrix, distCoeffs=None)

        """
            note:
                tvec is almost constant = the world origin coord with respect to the camera
                avarage value of tvec = [-1,-2,-21]
                we can use this directly without computing tvec
        """

        # convert rotation vector to rotation matrix .. note: function is used for vice versa
        # rotation matrix that transform from model coord(object model 3D space) to the camera 3D coord space
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # [R T] may be used in cv2.decomposeProjectionMatrix(extrinsic)[6]
        extrinsic_matrix = np.hstack((rotation_matrix, tvec))    
        
        # decompose the extrinsic matrix to many things including the 3 euler angles 
        # (pitch yaw roll) in degrees
        euler_angles = cv2.RQDecomp3x3(rotation_matrix)[0]

        return rvec, tvec, euler_angles

def draw_euler_axis(image, axis_pts, euler_angles):
    """
        draw euler axes in the image center 
    """
    center = (image.shape[1]//2, image.shape[0]//2)

    axis_pts = axis_pts.astype(np.int32)
    pitch_point = tuple(axis_pts[0].ravel())
    yaw_point   = tuple(axis_pts[1].ravel())
    roll_point  = tuple(axis_pts[2].ravel())

    pitch_color = (255,255,0)
    yaw_color   = (0,255,0)
    roll_color  = (0,0,255)

    pitch, yaw, roll = euler_angles

    cv2.line(image, center,  pitch_point, pitch_color, 2)
    cv2.line(image, center,  yaw_point, yaw_color, 2)
    cv2.line(image, center,  roll_point, roll_color, 2)
    cv2.putText(image, "Pitch:{:.2f}".format(pitch), (0,10), cv2.FONT_HERSHEY_PLAIN, 1, pitch_color)
    cv2.putText(image, "Yaw:{:.2f}".format(yaw), (0,20), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
    cv2.putText(image, "Roll:{:.2f}".format(roll), (0,30), cv2.FONT_HERSHEY_PLAIN, 1, roll_color)

    # origin
    cv2.circle(image, center, 2, (255,255,255), -1)
    return image

def draw_euler_angles(image, rvec, tvec, euler_angles, intrinsic_matrix):
        # i, j, k axes in world 3D coord.
        axis = np.identity(3) * 5
        # axis_img_pts = intrinsic * exstrinsic * axis
        axis_pts = cv2.projectPoints(axis, rvec, tvec, intrinsic_matrix, None)[0]
        image = draw_euler_axis(image, axis_pts, euler_angles)

        return image
        
def draw_annotation_box(image, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 255), line_width=2):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    rear_size = 10
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 10
    front_depth = 20
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      np.zeros((4, 1)))
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
        
def annotated_frame(frame, pupil_right_coords, pupil_left_coords):
    """Returns the main frame with pupils highlighted"""
    frame = frame.copy()

    color = (0, 255, 0)
    x_left, y_left = pupil_left_coords
    x_right, y_right = pupil_right_coords
    cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
    cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
    cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
    cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

    return frame

def gazeto3d(gaze):
    assert gaze.size == 2, "The size of gaze must be 2"
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt

if __name__ == "__main__":
    import sys, os
    import time
    import torch
    import torchvision.transforms as transforms
    sys.path.append(os.path.abspath('./modules/Tensorflow2.0-PFLD-'))
    from glue import PIPNet, ULFace
    from utils.landmark import *
    from utils.face_detection import *
    from copy import deepcopy

    import gazenet

    import utils_gaze
    
    from modules.OneEuroFilter import OneEuroFilter
    
    from scipy.spatial.transform import Rotation as R
    
    from model import Model

    landmark_predictor = PIPNet.Predictor("WFLW")
    face_detector = ULFace.Detector()
    

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('camera not detected')
    ret, frame = cap.read()
    euler_estimator = EulerAngles(frame.shape[:2])
    
    #GazeTR = Model()
    #modelpath = "/home/david/repos/Flux/"
    #statedict = torch.load(os.path.join(modelpath, "GazeTR-H-ETH.pt"), 
    #                       map_location={"cuda:0": "cuda:0"}
    #                      )
    #GazeTR.cuda(); GazeTR.load_state_dict(statedict); GazeTR.eval()

    net = gazenet.GazeNet("cuda:0")
    state_dict = torch.load("./weights/gazenet/gazenet.pth", map_location="cuda:0")
    net.load_state_dict(state_dict)

    net.eval()
    
    #filter_config_2d = {
    #    'freq': 30,        # system frequency about 30 Hz
    #    'mincutoff': 0.8,  # value refer to the paper
    #    'beta': 0.4,       # value refer to the paper
    #    'dcutoff': 0.4     # not mentioned, empirically set
    #}
    
    #filter_config_2d = {
    #    'freq': 30,        # system frequency about 30 Hz
    #    'mincutoff': 1,  # value refer to the paper
    #    'beta': 0.05,       # value refer to the paper
    #    'dcutoff': 1     # not mentioned, empirically set
    #}
    
    filter_config_2d = {
        'freq': 30,        # system frequency about 30 Hz
        'mincutoff': 0.8,  # value refer to the paper
        'beta': 0.4,       # value refer to the paper
        'dcutoff': 0.4     # not mentioned, empirically set
    }
    
    filter_2d = [(OneEuroFilter(**filter_config_2d),
                  OneEuroFilter(**filter_config_2d))
                  for _ in range(98)]
                  
    filter_rvec = (OneEuroFilter(**filter_config_2d),
                   OneEuroFilter(**filter_config_2d),
                   OneEuroFilter(**filter_config_2d))
    
    landmarks = np.empty( shape=(0, 0) )
    landmarks_for_drawing = np.empty( shape=(0, 0) )
    bbox = None
    bbox_prev = None
    last_detection = None
    is_face_detected = False
    
    rvec_cal = np.zeros((3,1))

    i = 0
    
    while cap.isOpened():
        ret, original = cap.read()
        frame = original.copy()
        if not ret: break

        height, width = frame.shape[:2]

        is_landmarks_detected = landmarks.size != 0

        is_face_detected, bboxes = face_detector.detect_bbox(frame)

        if is_face_detected:
            bbox = bboxes[0]
            bbox = face_detector.post_process(bbox)
            
            xmin, ymin, xmax, ymax = unwrap_bbox(bbox)
            
            cv2.rectangle(frame, (xmin, ymin),
                    (xmax, ymax), (125, 255, 0), 2)
            
            img = crop(frame, bbox)
            
            landmarks = landmark_predictor.predict(frame, bbox)
            
            landmarks_for_drawing = landmarks
            
            st = time.perf_counter()
            original = original[:,:,::-1]
            original = cv2.flip(original, 1)
            display = original.copy()
            face, gaze_origin, M  = utils_gaze.normalize_face(landmarks, original)
            print(time.perf_counter()-st)

            with torch.no_grad():
                gaze = net.get_gaze(face)
                gaze = gaze[0].data.cpu()
			
            display = cv2.circle(original, gaze_origin, 3, (0, 255, 0), -1)            
            display = utils_gaze.draw_gaze(original, gaze_origin, gaze, color=(255,0,0), thickness=2)         

            #gaze_img = cv2.resize(img, (224, 224))
            #with torch.no_grad():
            #
            #    transform = transforms.ToTensor()
            #    tensor = transform(gaze_img).unsqueeze(0).cuda()
            #
            #    gaze_img = {'face': tensor}
            #
            #    st =  time.perf_counter()
            #    gaze = GazeTR(gaze_img).detach().cpu().numpy()[0]
            #    print(time.perf_counter() - st)
            #
            #gaze_vec = gazeto3d(gaze)
            #print(gaze_vec)
            
            bbox_x_center = (bbox[0]+bbox[2])//2
            bbox_y_center = (bbox[1]+bbox[3])//2
            bbox_width = bbox[2]-bbox[0]
            
            eyes_x_center = int((landmarks_for_drawing[64][0]+landmarks_for_drawing[68][0])//2)
            eyes_y_center = int((landmarks_for_drawing[64][1]+landmarks_for_drawing[68][1])//2)
            
            #gaze_endpoint = (int(gaze_vec[0]*60+eyes_x_center), int(gaze_vec[1]*60+eyes_y_center))
            
            #cv2.circle(frame, gaze_endpoint, 15, (125, 255, 0))
            #cv2.line(frame, (eyes_x_center, eyes_y_center), gaze_endpoint, (125, 255, 0), 2)
            
            start_time = time.time()
            for j in range(98):
                #t = time.time()
                landmarks_for_drawing[j][0] = filter_2d[j][0](landmarks_for_drawing[j][0], time.time())
                landmarks_for_drawing[j][1] = filter_2d[j][1](landmarks_for_drawing[j][1], time.time())
            #print(time.time() - start_time)
            
            for (x, y) in landmarks_for_drawing:
                cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (125, 255, 0))
                
            #for point in [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]:
            for point in [46, 50, 38, 33, 72, 68, 64, 60, 59, 57, 55, 54, 82, 76, 79]:
                cv2.circle(frame, 
                            (np.int32(landmarks_for_drawing[point][0]), np.int32(landmarks_for_drawing[point][1])),
                             1, (0, 0, 255))
                
            rvec_detected, tvec, euler_angles = euler_estimator.euler_angles_from_landmarks(landmarks_for_drawing)
                
            #print(rvec)                
                
            rvec = rvec_detected - rvec_cal
            
            #print(rvec)
            
            for j in range(len(rvec)):
                #t = time.time()
                rvec[j] = filter_rvec[j](rvec[j], time.time())
            #print(rvec)
            #print()
            
            euler_angles = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0]
            
            calc_from_euler = R.from_euler('xyz', [euler_angles[0],euler_angles[1],euler_angles[2]], 
                                                   degrees=True).as_matrix()
                
            c_x = width / 2
            c_y = height / 2
            FieldOfView = 60
            focal = c_x / np.tan(np.radians(FieldOfView/2))
            
            camera_matrix = np.float32([[focal, 0.0,   c_x], 
                                 [0.0,   focal, c_y],
                                 [0.0,   0.0,   1.0]])
            
            draw_annotation_box(frame, rvec, tvec, camera_matrix, color=(0, 255, 0))

        cv2.imshow('1', display)
    
        i = i+1
    
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif ((k & 0xFF) == ord('c')) and is_face_detected:
            rvec_cal = deepcopy(rvec_detected)

    cap.release()
    cv2.destroyAllWindows()
