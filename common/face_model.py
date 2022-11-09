import dataclasses
import gc

import numpy as np
import pickle
import time
from scipy.spatial.transform import Rotation as R
from scipy.sparse import csr_matrix

from common.meshes import Model

import impasse

import torch
import pytorch3d
from pytorch3d.renderer.mesh import Textures
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.structures import Meshes


class SlothModel():
    @torch.no_grad()
    def __init__(self, gltf_path: str, device_str="cuda:0") -> None:
        self.device = torch.device(device_str)
        torch.cuda.empty_cache()

        assimp_scene = impasse.load(gltf_path)
        std_rot = R.from_euler("xyz", [90, 180, 0] ,degrees=True)
        self._std_rmat = torch.tensor(std_rot.as_matrix(), 
                                      dtype=torch.float32, 
                                      device=self.device)

        verts_neutral = np.array(assimp_scene.meshes[0].vertices, dtype=np.float32)*10
        self.verts_neutral = torch.tensor(verts_neutral, 
                                          device=self.device, 
                                          dtype=torch.float32)@self._std_rmat.T
        
        faces = np.array(assimp_scene.meshes[0].faces, dtype=np.float32)
        self.faces = torch.tensor(faces, device=self.device, dtype=torch.float32)

        self.bs_name_arr = np.array([target.name for target in assimp_scene.meshes[0].anim_meshes if 'Basis' not in target.name], dtype=str)
        
        rmat_np = self._std_rmat.T.cpu().numpy()
        target_arr = np.array([(target.vertices)@rmat_np for target in assimp_scene.meshes[0].anim_meshes if 'Basis' not in target.name])*10
        target_tensor = torch.tensor(target_arr, device=self.device, dtype=torch.float32)
        delta_tensor = target_tensor - self.verts_neutral

        x = delta_tensor.swapaxes(0, 2)[0]
        y = delta_tensor.swapaxes(0, 2)[1]
        z = delta_tensor.swapaxes(0, 2)[2]
        
        self.sparse_bs_x = x.to_sparse_csr()
        self.sparse_bs_y = y.to_sparse_csr()
        self.sparse_bs_z = z.to_sparse_csr()

        del assimp_scene

        io = IO()
        io.register_meshes_format(MeshGlbFormat())
        self.neutral_pt3d_mesh = io.load_mesh(gltf_path,
                                    device=self.device).update_padded(self.verts_neutral.unsqueeze(0))


    @torch.no_grad()
    def apply_weights_to_deltas(self, w):
        verts =  torch.vstack([torch.mv(self.sparse_bs_x, w.ravel()),
                               torch.mv(self.sparse_bs_y, w.ravel()),
                               torch.mv(self.sparse_bs_z, w.ravel())]).squeeze().T
        return verts

    @torch.no_grad()
    def apply_weights(self, w):
        verts =  self.verts_neutral + self.apply_weights_to_deltas(w)
        return verts

    @torch.no_grad()
    def apply_weights_to_mesh(self, w):
        mesh = self.neutral_pt3d_mesh.update_padded(self.apply_weights(w).unsqueeze(0))
        return mesh



class ICTFaceModel(Model):
    def __init__(self, ICT_dict: dict, load_blendshapes = True) -> None:
        neutral_vertices = ICT_dict['neutral']
        tri_faces = ICT_dict['topology']
        
        if load_blendshapes:
            blendshape_names = list(ICT_dict['deltas'].keys())
            blendshape_arr = np.array(list(ICT_dict['deltas'].values()))
            super().__init__(neutral_vertices, tri_faces, blendshape_arr, blendshape_names)
        else:
            super().__init__(neutral_vertices, tri_faces)
        

    @classmethod
    def from_pkl(cls, pkl_path: str, load_blendshapes = True):
        with open(pkl_path, "rb") as ICT_file:
            ICT_dict = pickle.load(ICT_file)
        return cls(ICT_dict, load_blendshapes)


class ICTFaceModel68(Model):
    def __init__(self, ICT_dict: dict, load_blendshapes = True, store_bs_dict = False) -> None:
        self.sparse_68_idxs = [1225,1888,1052,367,1719,1722,2199,1447,966,3661,4390,3927,3924,2608,
                                3272,4088,3443,268,493,1914,2044,1401,3615,4240,4114,2734,2509,978,
                                4527,4942,4857,1140,2075,1147,4269,3360,1507,1542,1537,1528,1518,
                                1511,3742,3751,3756,3721,3725,3732,5708,5695,2081,0,4275,6200,6213,
                                6346,6461,5518,5957,5841,5702,5711,5533,6216,6207,6470,5517,5966]

        neutral_vertices = ICT_dict['neutral'][self.sparse_68_idxs]
        tri_faces = ICT_dict['topology']

        if load_blendshapes:
            if store_bs_dict:
                self.bs_dict = ICT_dict['deltas']
            blendshape_names = list(ICT_dict['deltas'].keys())
            blendshape_arr = np.array(list(ICT_dict['deltas'].values()))[:,self.sparse_68_idxs]
            super().__init__(neutral_vertices, tri_faces, blendshape_arr, blendshape_names)
        else:
            super().__init__(neutral_vertices, tri_faces)

    @classmethod
    def from_pkl(cls, pkl_path: str, load_blendshapes = True, store_bs_dict = False):
        with open(pkl_path, "rb") as ICT_file:
            ICT_dict = pickle.load(ICT_file)
        return cls(ICT_dict, load_blendshapes, store_bs_dict)



class ICTModelPT3D():
    @torch.no_grad()
    def __init__(self, pkl_path: str, device_str="cuda:0") -> None:
        self.device = torch.device(device_str)
        torch.cuda.empty_cache()

        ICT_model = ICTFaceModel.from_pkl(pkl_path, load_blendshapes = True)

        std_rot = R.from_euler("xyz", [0, 0, 180] ,degrees=True)
        self._std_rmat = torch.tensor(std_rot.as_matrix(), 
                                      dtype=torch.float32, 
                                      device=self.device)

        self.verts_neutral = torch.tensor(ICT_model.neutral_vertices, 
                                          device=self.device, 
                                          dtype=torch.float32)@self._std_rmat.T
        
        self.faces = torch.tensor(ICT_model.faces, device=self.device, dtype=torch.float32)

        self.bs_name_arr = np.array(ICT_model.bs_names, dtype=str)
        
        rmat_np = self._std_rmat.T.cpu().numpy()
        delta_arr = ICT_model.get_blendshape_arr()
        delta_tensor = torch.tensor(delta_arr@rmat_np, device=self.device, dtype=torch.float32)

        x = delta_tensor.swapaxes(0, 2)[0]
        y = delta_tensor.swapaxes(0, 2)[1]
        z = delta_tensor.swapaxes(0, 2)[2]
        
        self.sparse_bs_x = x.to_sparse_csr()
        self.sparse_bs_y = y.to_sparse_csr()
        self.sparse_bs_z = z.to_sparse_csr()

        del ICT_model

        io = IO()
        verts_rgb = torch.ones_like(self.verts_neutral)[None]
        textures = Textures(verts_rgb=verts_rgb.to(self.device))
        self.neutral_pt3d_mesh =  Meshes( verts=[self.verts_neutral.to(self.device)], faces=[self.faces.to(self.device)], textures=textures)


    @torch.no_grad()
    def apply_weights_to_deltas(self, w):
        verts =  torch.vstack([torch.mv(self.sparse_bs_x, w.ravel()),
                               torch.mv(self.sparse_bs_y, w.ravel()),
                               torch.mv(self.sparse_bs_z, w.ravel())]).squeeze().T
        return verts

    @torch.no_grad()
    def apply_weights(self, w):
        verts =  self.verts_neutral + self.apply_weights_to_deltas(w)
        return verts

    @torch.no_grad()
    def apply_weights_to_mesh(self, w):
        mesh = self.neutral_pt3d_mesh.update_padded(self.apply_weights(w).unsqueeze(0))
        return mesh



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
        
        #s = time.perf_counter()
        transformed_model = (rotated_model-[model_eyes_midpoint[0], model_eyes_midpoint[1], 0])*scale+[pred_eyes_midpoint[0], pred_eyes_midpoint[1], 0]
        #print("t==", time.perf_counter()-s)

        return transformed_model
