import numpy as np
import cv2
import time
import pickle

from common.meshes import Model

from jaxfit import CurveFit
import jax.numpy as jnp
from jax import jit


class ExpressionFitting:
    def __init__(self) -> None:
        with open("./common/ICTFaceModel.pkl", "rb") as ICT_file:
            ICT_Model = pickle.load(ICT_file)
            ICT_to_iBUG_idxs = [1225,1888,1052,367,1719,1722,2199,1447,966,3661,4390,
                                3927,3924,2608,3272,4088,3443,268,493,1914,2044,1401,
                                3615,4240,4114,2734,2509,978,4527,4942,4857,1140,2075,
                                1147,4269,3360,1507,1542,1537,1528,1518,1511,3742,3751,
                                3756,3721,3725,3732,5708,5695,2081,0,4275,6200,6213,6346,
                                6461,5518,5957,5841,5702,5711,5533,6216,6207,6470,5517,5966]
        self.neutral = ICT_Model['neutral'][ICT_to_iBUG_idxs]
        self.triangles = 

        pass