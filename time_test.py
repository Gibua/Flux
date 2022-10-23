import sys, os
import time
import math
import cv2
import numpy as np
np.set_printoptions(suppress=True)
import copy

from decimal import Decimal as D

import gc


import torch
torch_device = torch.device("cuda:0")
torch.cuda.set_device(torch_device)
torch.cuda.set_per_process_memory_fraction(0.2, device=torch_device)

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
#tf.config.experimental.set_visible_devices([], "GPU")

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128'

from scipy.spatial.transform import Rotation as R

from common.landmark_mapping import LandmarkMapper
from common.mappings import Datasets
from common.camera import PinholeCamera
from common.fitting import ExpressionFitting
from common.head_pose import PoseEstimator2D, draw_axes, draw_angles_text, draw_annotation_box
from common.face_model import ICTFaceModel68, SlothModel, ICTModelPT3D

from modules.OneEuroFilter import OneEuroFilter

from utils.landmark import *
from utils.face_detection import *
from utils.mesh import generate_w_mapper

from VideoGet import VideoGet
import gc

from glue import RetinaFace, PFLD_UltraLight, PFLD_TFLite, ULFace, RetinaFace, SynergyNet, PIPNet



import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (PointLights,
                                RasterizationSettings,
                                MeshRenderer,
                                MeshRasterizer,
                                HardPhongShader)
from pytorch3d.renderer.cameras import PerspectiveCameras, FoVPerspectiveCameras


def pt3d_render(mesh, camera: PinholeCamera, width, height, rvec = None, tvec = None):
    device = mesh.device
    if rvec is None:
        rmat=torch.tensor(np.eye(3), device=device).unsqueeze(0)
    else:
        euler_angles = R.from_rotvec(rvec.ravel()).as_euler('XYZ', degrees=True)
        rmat_np = R.from_euler('XYZ', euler_angles*[1,1,-1], degrees=True).as_matrix()
        rmat = torch.tensor(rmat_np, device=device).unsqueeze(0)

    if tvec is None:
        tvec_tensor = torch.reshape(torch.tensor([0,0,4], device=device), (1,3))
    else:
        tvec_tensor = torch.reshape(torch.tensor( (tvec.ravel()*[-0.5,-0.5,0.5]), device=device), (1,3))

    p_point = torch.tensor(camera.get_center(), device=device).unsqueeze(0)
    f = torch.tensor(camera.get_focal(), device=device).unsqueeze(0)
    img_size = torch.tensor((height, width), device=device).unsqueeze(0)

    #cameras = PerspectiveCameras(R=rmat, T=tvec_tensor, device=device, image_size= img_size)
    cameras = FoVPerspectiveCameras(device=device, R=rmat, T=tvec_tensor)
    
    raster_settings = RasterizationSettings(
        image_size=(height, width), 
        blur_radius=0.0, 
        faces_per_pixel=1)

    lights = PointLights(device=device, location=[[20, 10, -40.0]])
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings),
        shader=HardPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    images = renderer(mesh, device=device)

    #plt.figure(figsize=(10, 10)).set_facecolor("k")
    #plt.imshow(images[0, ..., :3].cpu().numpy())
    #plt.axis("off");

    return cv2.cvtColor(images[0, ..., :3].cpu().numpy(), cv2.COLOR_BGR2RGB)


def orthoProjection(points: np.ndarray, rvec: np.ndarray, tx, ty, scale):
    rmat = cv2.Rodrigues(rvec)[0]
    translation = np.array([tx, ty, 0])
    projected = ((points.copy()*scale).dot(rmat.T) + translation)

    return projected


def putTextCenter(img, text: str, center, fontFace, fontScale: int, color, thickness: int):
    textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

    center_x = np.int32(center[0] - (textsize[0]/2.))
    center_y = np.int32(center[1] + (textsize[1]/2.))

    cv2.putText(img, text, (center_x, center_y), fontFace, fontScale, color, thickness)


#if __name__ == "__main__":
def main(video_getter, face_detector, landmark_predictor):
    ret, frame = video_getter.read()
    height, width = frame.shape[:2]

    #landmark_predictor = PIPNet.Predictor("WFLW")
    
    dataset = landmark_predictor.dataset
    #landmarks_n = landmark_predictor.landmark_count
    landmarks_n = 68
    if dataset == Datasets.WFLW:
        mapper = LandmarkMapper(Datasets.WFLW, Datasets.IBUG)

    cam = PinholeCamera(width, height)#, camera_matrix=s_cmat)

    hp_estimator = PoseEstimator2D(cam)

    ICT_Model = ICTFaceModel68.from_pkl("./common/ICTFaceModel.pkl")
    
    retarg_model = SlothModel('/home/david/repos/Flux/common/sloth_scaled.glb')
    retarg_model_2 = ICTModelPT3D("./common/ICTFaceModel.pkl")

    bs_mapper = generate_w_mapper(ICT_Model.bs_names, retarg_model.bs_name_arr.tolist())#, use_jax=True)

    fitter = ExpressionFitting(cam.camera_matrix)#, bs_mapper = bs_mapper)

    #filter_config_2d = {
    #    'freq': 30,
    #    'mincutoff': 0.8,
    #    'beta': 0.4,
    #    'dcutoff': 0.4 
    #}

    #filter_config_2d = {
    #    'freq': 30,
    #    'mincutoff': 1,
    #    'beta': 0.05,
    #    'dcutoff': 1
    #}

    filter_config = {
        'freq': 30,        # Values from: https://github.com/XinArkh/VNect/ ([Mehta et al. 2017])
        'mincutoff': 0.4,  # Left the same because those values give good results, empirically
        'beta': 0.6,       # 
        'dcutoff': 0.4     # 
    }

    filter_2d = [(OneEuroFilter(**filter_config),
                  OneEuroFilter(**filter_config))
                  for _ in range(landmarks_n)]

    filter_rvec = (OneEuroFilter(**filter_config),
                   OneEuroFilter(**filter_config),
                   OneEuroFilter(**filter_config))
    
    landmarks = np.empty( shape=(0, 0) )
    bbox = None
    bbox_prev = None
    last_detection = None
    is_face_detected = False
    w = None
    n_its = 0
    last_time = time.time()
    model_img = np.full((height, width, 3), 64, dtype=np.uint8)
    model_img_2 = np.full((height, width, 3), 64, dtype=np.uint8)

    #b_fit    = D('2.8')
    #b_prior  = D('15.6')
    #b_sparse = D('8.5')

    b_fit    = D('2.20')
    b_prior  = D('15.20')
    b_sparse = D('9.30')

    l_step = D('0.10')
    s_step = D('0.01')

    b_trans_lim = D('0.3')
    max_b = D('5')

    captures_total = 0
    intervals_list = []

    while video_getter.cap.isOpened():
        if captures_total%100 ==0:
            print(captures_total, end = ' ')
        curr_start = time.perf_counter()
        ret, original = video_getter.read()
        if not ret: break
        frame = original.copy()
        
        height, width = frame.shape[:2]

        model_img = np.full((height, width, 3), 64, dtype=np.uint8)

        is_landmarks_detected = landmarks.size != 0
        
        time_elapsed = time.time()-last_time
        if (n_its == 0) or (time_elapsed > 2.5) or (not is_face_detected):
            last_time = time.time()
            is_face_detected, bboxes = face_detector.detect_bbox(frame)
            if is_face_detected and (not is_landmarks_detected):
                last_detection = bboxes[0]
                bbox = last_detection
                bbox_prev = last_detection
        if (n_its > 0) and is_face_detected and is_landmarks_detected:
            r_corner, l_corner = (45, 36)
            landmark_bbox = bbox_from_landmark(landmarks, r_corner, l_corner)

            intersection = bbox_intersect(last_detection, landmark_bbox)

            landmark_bbox_area = bbox_area(landmark_bbox)
            last_detection_area = bbox_area(last_detection)
            intersect_area = bbox_area(intersection)
            intersect_proportion = intersect_area/max(landmark_bbox_area,last_detection_area)

            if (intersect_proportion<0.50):
                is_face_detected, bboxes = face_detector.detect_bbox(frame)
                if is_face_detected:
                    last_detection = bboxes[0]
                    bbox = last_detection
                    bbox_prev = last_detection
            else:
                bbox_prev = bbox
                bbox = bboxes_average(landmark_bbox, bbox_prev)
        
        if is_face_detected:
            #bbox = face_detector.post_process(bbox)
            bbox = crop_at_corners(bbox, width, height).astype(int)

            if landmark_predictor.pose_is_provided:
                det_landmarks, det_rvec, det_tvec = landmark_predictor.predict(frame, bbox)
            else:
                det_landmarks = landmark_predictor.predict(frame, bbox)
            landmarks = det_landmarks[:,0:2]

            if dataset not in [Datasets.IBUG, Datasets.AFLW3D]:
                landmarks = mapper.map_landmarks(landmarks)
            
            #xmin, ymin, xmax, ymax = unwrap_bbox(bbox)
            #cv2.rectangle(frame, (xmin, ymin),
            #                     (xmax, ymax), (125, 255, 0), 2)
            
            for j in range(landmarks_n):
                #t = time.time()
                landmarks[j][0] = filter_2d[j][0](landmarks[j][0], time.time())
                landmarks[j][1] = filter_2d[j][1](landmarks[j][1], time.time())
            
            #for (x, y) in landmarks:
            #    cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (125, 255, 0))
            
            rvec, tvec = hp_estimator.solve_pose(landmarks, True)
            
            #pitch_color = (210,200,0)
            #yaw_color   = (50,150,0)
            #roll_color  = (0,0,255)
            #cv2.putText(frame, "b_fit:{:.2f}".format(float(b_fit)), (0,10+45), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
            #cv2.putText(frame, "b_prior:{:.2f}".format(float(b_prior)), (0,25+45), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
            #cv2.putText(frame, "b_sparse:{:.2f}".format(float(b_sparse)), (0,40+45), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
            
            #draw_angles_text(frame, rvec)
            #draw_annotation_box(frame, rvec, tvec, cam)
            #draw_axes(frame, rvec, tvec, cam.camera_matrix)#, scale = 1000)

            w_dlp = fitter.fit(landmarks, rvec, tvec, float(b_fit), float(b_prior), float(b_sparse), method = 'jaxopt_lm')#, debug=True)

            #intervals_list.append(time.perf_counter() - curr_start)
            #captures_total += 1
            #print(np.from_dlpack(w_dlp))
            #w_tensor = torch.from_dlpack(w_dlp).float()
            #print(w_tensor.cpu().numpy())

            #mesh_2_weighted = retarg_model_2.apply_weights_to_mesh(w_tensor)
            #model_img_2 = pt3d_render(mesh_2_weighted, cam, width, height, rvec = rvec, tvec = tvec)

            #w_tensor_sloth = bs_mapper(w_tensor)
            #
            #mesh_weighted = retarg_model.apply_weights_to_mesh(w_tensor_sloth)
            #
            #model_img = pt3d_render(mesh_weighted, cam, width, height, rvec = rvec, tvec = tvec)
            #
            ##weighted = mesh_weighted.verts_packed().cpu().numpy()
            ##faces =  mesh_weighted.faces_list()[0].cpu().numpy()

            #ict_weighted = ICT_Model.apply_weights(w_np)
            #projected = cam.project_points(ict_weighted, rvec, tvec)#, scale=15)
            #for (x, y) in projected:
            #    cv2.circle(frame, (np.int32(x), np.int32(y)), 1, (125, 255, 200))
            #
            ##cv2.polylines(frame, projected[:,:2][ICT_Model.faces].astype(int), True, (210, 190, 190), 1, cv2.LINE_AA)

            #hp_estimator.project_model(rvec, tvec)

        #cv2.imshow('features', frame)
        #cv2.imshow('retarget', model_img)
        #cv2.imshow('retarget 2', model_img_2)

        n_its += 1
        
        k = cv2.pollKey()
        
        if k == 27:
            break
        if ((k & 0xFF) == ord('c')) and is_face_detected:
            hp_estimator.set_calibration(landmarks)
            #hp_estimator.set_calibration(rvec)
        if ((k & 0xFF) == ord('q')):
            if (b_fit<max_b) and (b_fit>=D('0')):
                if b_fit >= b_trans_lim:
                    b_fit += l_step
                else:
                    b_fit += s_step
        if ((k & 0xFF) == ord('w')):
            if (b_fit<=max_b) and (b_fit>D('0')):
                if b_fit > b_trans_lim:
                    b_fit -= l_step
                else:
                    b_fit -= s_step

        if ((k & 0xFF) == ord('e')):
            if (b_prior<D('40')) and (b_prior>=D('0')):
                if b_prior >= b_trans_lim:
                    b_prior += l_step
                else:
                    b_prior += s_step
        if ((k & 0xFF) == ord('r')):
            if (b_prior<=D('40')) and (b_prior>D('0')):
                if b_prior > b_trans_lim:
                    b_prior -= l_step
                else:
                    b_prior -= s_step

        if ((k & 0xFF) == ord('t')):
            if (b_sparse<D('10')) and (b_sparse>=D('0')):
                if b_sparse >= b_trans_lim:
                    b_sparse += l_step
                else:
                    b_sparse += s_step
        if ((k & 0xFF) == ord('y')):
            if (b_sparse<=D('10')) and (b_sparse>D('0')):
                if b_sparse > b_trans_lim:
                    b_sparse -= l_step
                else:
                    b_sparse -= s_step
        
        intervals_list.append(time.perf_counter() - curr_start)
        captures_total += 1
        #if captures_total%100 ==0:
        #print(captures_total, end = ' ')
        if captures_total >= 1000:
            avg = sum(intervals_list)/captures_total
            return avg


if __name__ == "__main__":
    import traceback

    video_getter = VideoGet(src=0).start()
    if not video_getter.cap.isOpened():
        print('camera not detected')
        video_getter.stop()
        video_getter.cap.release()
        sys.exit()
    else:
        try:
            for f_detector in ['RetinaFace', 'ULFace']:
                if f_detector == 'RetinaFace':
                    face_detector = RetinaFace.Detector()
                elif f_detector == 'ULFace':
                    face_detector = ULFace.Detector()
                
                for lm_detector in ['PFLD_UltraLight', 'PFLD_TFLite', 'SynergyNet', 'PIPNet_WFLW', 'PIPNet_300W']:
                    if lm_detector == 'PFLD_UltraLight':
                        landmark_predictor = PFLD_UltraLight.Predictor()
                    elif lm_detector == 'PFLD_TFLite':
                        landmark_predictor = PFLD_TFLite.Predictor()
                    elif lm_detector == 'SynergyNet':
                        landmark_predictor = SynergyNet.Predictor()
                    elif lm_detector == 'PIPNet_WFLW':
                        landmark_predictor = PIPNet.Predictor("WFLW")
                    elif lm_detector == 'PIPNet_300W':
                        landmark_predictor = PIPNet.Predictor('300W_COFW_WFLW')

                    time_ret = main(video_getter, face_detector, landmark_predictor)

                    print(f_detector, 'x', lm_detector, '=', time_ret)

                    del landmark_predictor

                del face_detector
        except KeyboardInterrupt:
            print("Exiting...")
        except Exception as e:
            #print(str(type(e).__name__)+':',e)
            traceback.print_exc()
            
    video_getter.stop()
    video_getter.cap.release()
    cv2.destroyAllWindows()
    sys.exit()