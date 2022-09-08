import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation as R

@dataclasses.dataclass(frozen=True)
class FaceModel68():
    """3D face model for Multi-PIE 68 points mark-up.

    In the camera coordinate system, the X axis points to the right from
    camera, the Y axis points down, and the Z axis points forward.

    The face model is facing the camera. Here, the Z axis is
    perpendicular to the plane passing through the three midpoints of
    the eyes and mouth, the X axis is parallel to the line passing
    through the midpoints of both eyes, and the origin is at the tip of
    the nose.

    The units of the coordinate system are meters and the distance
    between outer eye corners of the model is set to 90mm.

    The model coordinate system is defined as the camera coordinate
    system rotated 180 degrees around the Y axis.
    """
    LANDMARKS = np.array([
        [-0.07141807, -0.02827123, 0.08114384],  #  0
        [-0.07067417, -0.00961522, 0.08035654],  #  1
        [-0.06844646,  0.00895837, 0.08046731],  #  2
        [-0.06474301,  0.02708319, 0.08045689],  #  3
        [-0.05778475,  0.04384917, 0.07802191],  #  4
        [-0.04673809,  0.05812865, 0.07192291],  #  5
        [-0.03293922,  0.06962711, 0.06106274],  #  6
        [-0.01744018,  0.07850638, 0.04752971],  #  7
        [ 0.        ,  0.08105961, 0.0425195 ],  #  8
        [ 0.01744018,  0.07850638, 0.04752971],  #  9
        [ 0.03293922,  0.06962711, 0.06106274],  # 10
        [ 0.04673809,  0.05812865, 0.07192291],  # 11
        [ 0.05778475,  0.04384917, 0.07802191],  # 12
        [ 0.06474301,  0.02708319, 0.08045689],  # 13
        [ 0.06844646,  0.00895837, 0.08046731],  # 14
        [ 0.07067417, -0.00961522, 0.08035654],  # 15
        [ 0.07141807, -0.02827123, 0.08114384],  # 16
        [-0.05977758, -0.0447858 , 0.04562813],  # 17  right eyebrow - right corner
        [-0.05055506, -0.05334294, 0.03834846],  # 18
        [-0.0375633 , -0.05609241, 0.03158344],  # 19
        [-0.02423648, -0.05463779, 0.02510117],  # 20
        [-0.01168798, -0.04986641, 0.02050337],  # 21  right eyebrow - left corner
        [ 0.01168798, -0.04986641, 0.02050337],  # 22  left eyebrow - right corner
        [ 0.02423648, -0.05463779, 0.02510117],  # 23
        [ 0.0375633 , -0.05609241, 0.03158344],  # 24
        [ 0.05055506, -0.05334294, 0.03834846],  # 25
        [ 0.05977758, -0.0447858 , 0.04562813],  # 26  left eyebrow - left corner
        [ 0.        , -0.03515768, 0.02038099],  # 27
        [ 0.        , -0.02350421, 0.01366667],  # 28
        [ 0.        , -0.01196914, 0.00658284],  # 29
        [ 0.        ,  0.        , 0.        ],  # 30  nose - tip
        [-0.01479319,  0.00949072, 0.01708772],  # 31  nose - right corner
        [-0.00762319,  0.01179908, 0.01419133],  # 32
        [ 0.        ,  0.01381676, 0.01205559],  # 33  nose - center (columella / philtrum)
        [ 0.00762319,  0.01179908, 0.01419133],  # 34
        [ 0.01479319,  0.00949072, 0.01708772],  # 35  nose - left corner
        [-0.045     , -0.032415  , 0.03976718],  # 36  right eye - right corner
        [-0.0370546 , -0.0371723 , 0.03579593],  # 37
        [-0.0275166 , -0.03714814, 0.03425518],  # 38
        [-0.01919724, -0.03101962, 0.03359268],  # 39  right eye - left corner
        [-0.02813814, -0.0294397 , 0.03345652],  # 40
        [-0.03763013, -0.02948442, 0.03497732],  # 41
        [ 0.01919724, -0.03101962, 0.03359268],  # 42  left eye - right corner
        [ 0.0275166 , -0.03714814, 0.03425518],  # 43
        [ 0.0370546 , -0.0371723 , 0.03579593],  # 44
        [ 0.045     , -0.032415  , 0.03976718],  # 45  left eye - left corner
        [ 0.03763013, -0.02948442, 0.03497732],  # 46
        [ 0.02813814, -0.0294397 , 0.03345652],  # 47
        [-0.02847002,  0.03331642, 0.03667993],  # 48
        [-0.01796181,  0.02843251, 0.02335485],  # 49
        [-0.00742947,  0.0258057 , 0.01630812],  # 50
        [ 0.        ,  0.0275555 , 0.01538404],  # 51
        [ 0.00742947,  0.0258057 , 0.01630812],  # 52
        [ 0.01796181,  0.02843251, 0.02335485],  # 53
        [ 0.02847002,  0.03331642, 0.03667993],  # 54
        [ 0.0183606 ,  0.0423393 , 0.02523355],  # 55
        [ 0.00808323,  0.04614537, 0.01820142],  # 56
        [ 0.        ,  0.04688623, 0.01716318],  # 57
        [-0.00808323,  0.04614537, 0.01820142],  # 58
        [-0.0183606 ,  0.0423393 , 0.02523355],  # 59
        [-0.02409981,  0.03367606, 0.03421466],  # 60
        [-0.00756874,  0.03192644, 0.01851247],  # 61
        [ 0.        ,  0.03263345, 0.01732347],  # 62
        [ 0.00756874,  0.03192644, 0.01851247],  # 63
        [ 0.02409981,  0.03367606, 0.03421466],  # 64
        [ 0.00771924,  0.03711846, 0.01940396],  # 65
        [ 0.        ,  0.03791103, 0.0180805 ],  # 66
        [-0.00771924,  0.03711846, 0.01940396],  # 67
    ],
                                     dtype=np.float64)

    #LANDMARKS = np.array([
    #    [-72.3846726 , -28.65386488, 82.24207532],  #  0
    #    [-71.63070434,  -9.74535649, 81.44411966],  #  1
    #    [-69.37284356,   9.0796164 , 81.55638887],  #  2
    #    [-65.61926949,  27.44974544, 81.54582784],  #  3
    #    [-58.56683344,  44.44264336, 79.07789179],  #  4
    #    [-47.37066323,  58.91538793, 72.89634532],  #  5
    #    [-33.38503344,  70.56947298, 61.88918915],  #  6
    #    [-17.67622283,  79.56891881, 48.17299735],  #  7
    #    [  0.0       ,  82.15670531, 43.09497704],  #  8
    #    [ 17.67622283,  79.56891881, 48.17299735],  #  9
    #    [ 33.38503344,  70.56947298, 61.88918915],  # 10
    #    [ 47.37066323,  58.91538793, 72.89634532],  # 11
    #    [ 58.56683344,  44.44264336, 79.07789179],  # 12
    #    [ 65.61926949,  27.44974544, 81.54582784],  # 13
    #    [ 69.37284356,   9.0796164 , 81.55638887],  # 14
    #    [ 71.63070434,  -9.74535649, 81.44411966],  # 15
    #    [ 72.3846726 , -28.65386488, 82.24207532],  # 16
    #    [-60.58663525, -45.39195011, 46.24568056],  # 17  right eyebrow - right corner
    #    [-51.23929373, -54.06490609, 38.8674844 ],  # 18
    #    [-38.07169771, -56.85158859, 32.01090374],  # 19
    #    [-24.56450685, -55.37728114, 25.44089993],  # 20
    #    [-11.8461701 , -50.54132325, 20.78087135],  # 21  right eyebrow - left corner
    #    [ 11.8461701 , -50.54132325, 20.78087135],  # 22  left eyebrow - right corner
    #    [ 24.56450685, -55.37728114, 25.44089993],  # 23
    #    [ 38.07169771, -56.85158859, 32.01090374],  # 24
    #    [ 51.23929373, -54.06490609, 38.8674844 ],  # 25
    #    [ 60.58663525, -45.39195011, 46.24568056],  # 26  left eyebrow - left corner
    #    [  0.0       , -35.63351903, 20.65683501],  # 27
    #    [  0.0       , -23.822326  , 13.85164054],  # 28
    #    [  0.0       , -12.13113544,  6.67193496],  # 29
    #    [  0.0       ,   0.0       ,  0.0       ],  # 30  nose - tip
    #    [-14.99340734,   9.61917145, 17.31899249],  # 31  nose - right corner
    #    [ -7.7263655 ,  11.95877378, 14.38340151],  # 32
    #    [  0.0       ,  14.00376192, 12.21875549],  # 33  nose - center (columella / phil
    #    [  7.7263655 ,  11.95877378, 14.38340151],  # 34
    #    [ 14.99340734,   9.61917145, 17.31899249],  # 35  nose - left corner
    #    [-45.60904918, -32.85371843, 40.30540596],  # 36  right eye - right corner
    #    [-37.55611275, -37.67540575, 36.28040737],  # 37
    #    [-27.88902139, -37.65091876, 34.71880421],  # 38
    #    [-19.45706363, -31.43945276, 34.04733765],  # 39  right eye - left corner
    #    [-28.51897358, -29.83814945, 33.9093348 ],  # 40
    #    [-38.13943222, -29.88347471, 35.45071796],  # 41
    #    [ 19.45706363, -31.43945276, 34.04733765],  # 42  left eye - right corner
    #    [ 27.88902139, -37.65091876, 34.71880421],  # 43
    #    [ 37.55611275, -37.67540575, 36.28040737],  # 44
    #    [ 45.60904918, -32.85371843, 40.30540596],  # 45  left eye - left corner
    #    [ 38.13943222, -29.88347471, 35.45071796],  # 46
    #    [ 28.51897358, -29.83814945, 33.9093348 ],  # 47
    #    [-28.85534539,  33.76733863, 37.17637181],  # 48
    #    [-18.20491279,  28.81732771, 23.67094449],  # 49
    #    [ -7.53002361,  26.15496534, 16.52884105],  # 50
    #    [  0.0       ,  27.92844788, 15.59225415],  # 51
    #    [  7.53002361,  26.15496534, 16.52884105],  # 52
    #    [ 18.20491279,  28.81732771, 23.67094449],  # 53
    #    [ 28.85534539,  33.76733863, 37.17637181],  # 54
    #    [ 18.60910019,  42.91233813, 25.57507162],  # 55
    #    [  8.19263188,  46.76992111, 18.44776578],  # 56
    #    [  0.0       ,  47.52080822, 17.39547379],  # 57
    #    [ -8.19263188,  46.76992111, 18.44776578],  # 58
    #    [-18.60910019,  42.91233813, 25.57507162],  # 59
    #    [-24.4259871 ,  34.13184615, 34.67773579],  # 60
    #    [ -7.67117855,  32.35854605, 18.76302566],  # 61
    #    [  0.0       ,  33.07512502, 17.55793323],  # 62
    #    [  7.67117855,  32.35854605, 18.76302566],  # 63
    #    [ 24.4259871 ,  34.13184615, 34.67773579],  # 64
    #    [  7.82371548,  37.62083706, 19.66658147],  # 65
    #    [  0.0       ,  38.42413404, 18.32520919],  # 66
    #    [ -7.82371548,  37.62083706, 19.66658147],  # 67
    #    ], 
    #    dtype=np.float64)


    REYE_INDICES = np.array([36, 39])
    LEYE_INDICES = np.array([42, 45])
    MOUTH_INDICES = np.array([48, 54])
    NOSE_INDICES = np.array([31, 35])

    CHIN_INDEX = 8
    NOSE_INDEX = 30

    @staticmethod
    def transform_model(rvec, landmarks_2D):
        model_rot = R.from_rotvec(rvec.flatten())
        rotated_model = model_rot.apply(FaceModel68.LANDMARKS.copy())
        
        #pred_l_eye_center = np.mean(landmarks_2D[np.array([36, 39])], axis=0)
        #pred_r_eye_center = np.mean(landmarks_2D[np.array([42, 45])], axis=0)
        #pred_eye_center = np.mean([pred_l_eye_center, pred_r_eye_center], axis=0)
        #pred_eye_distance = np.linalg.norm(pred_r_eye_center - pred_l_eye_center)
        pred_eyes_midpoint = np.mean([landmarks_2D[36], landmarks_2D[45]], axis=0) #  outer eye indexes = [36, 45]
        pred_eye_2D_distance = np.linalg.norm(landmarks_2D[45] - landmarks_2D[36])

        #model_2d = rotated_model[:,0:1+1]
        
        #model_l_eye_center = np.mean(model_2d[np.array([36, 39])], axis=0)
        #model_r_eye_center = np.mean(model_2d[np.array([42, 45])], axis=0)
        #model_eye_center = np.mean([model_l_eye_center, model_r_eye_center], axis=0)
        model_eyes_midpoint = np.mean([rotated_model[36], rotated_model[45]], axis=0)
        model_eye_2D_distance = np.linalg.norm(rotated_model[45][:2] - rotated_model[36][:2])

        #print(model_eyes_midpoint, pred_eyes_midpoint)
        
        scale = pred_eye_2D_distance/model_eye_2D_distance
        
        transformed_model = (rotated_model-np.float64([model_eyes_midpoint[0], model_eyes_midpoint[1], 0]))*scale+np.float64([pred_eyes_midpoint[0], pred_eyes_midpoint[1], 0])

        return transformed_model