from functools import partial
import numpy as np
import cv2
import os
import sys
import time
import pickle
from scipy.optimize import least_squares, minimize, Bounds, curve_fit
from lmfit import Parameters, fit_report, minimize

from common.meshes import Model
from common.face_model import ICTFaceModel68

import jax
from jax.config import config
#config.update("jax_debug_nans", True) 
config.update("jax_enable_x64", True)
#os.environ['JAX_PLATFORMS']='cpu'

import jax.dlpack as jdlp

from jax import jit
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
from jaxopt.linear_solve import solve_cg

from jaxopt import LevenbergMarquardt, LBFGS, ScipyBoundedMinimize, ScipyBoundedLeastSquares, ScipyLeastSquares

#@partial(jax.jit, static_argnames=['neutral', 'bs_arr'])


def unpack(params):
    measured_points = params[0:68,0:2]
    rmat = params[68:68+3,:]
    tvec = params[68+3]
    return measured_points,rmat,tvec


def project_points(points, rmat, tvec, cmat):
    transformed = (points@rmat.T + tvec.reshape(-1))@cmat.T
    projected = jnp.divide(transformed[:,:2], transformed[:,2:])
    return projected

def ortho_proj(points, rmat, tvec, cmat):
    scale = jnp.divide(cmat[0][0], tvec.ravel()[2])
    c_x = cmat[0][2]
    c_y = cmat[1][2]
    t_x = c_x + tvec.ravel()[0]*scale
    t_y = c_y + tvec.ravel()[1]*scale
    translation = jnp.array([t_x, t_y, 0])
    projected = ((points*scale).dot(rmat.T) + translation)[:,2]
    return projected


#def residuals(w, measured_points, rmat, tvec, cmat, neutral, bs_arr):
@jax.jit
def residuals(w, non_fit_params, cmat, former_w, neutral, bs_arr):
    b_fit = 0.3
    b_sparse = 1
    b_prior = 1

    measured_points, rmat, tvec = unpack(non_fit_params)     

    sparse_weighted = neutral+(bs_arr.T@w.reshape(-1)).T
    
    e_fit = project_points(sparse_weighted, rmat, tvec, cmat) - measured_points # face it

    #hcb.id_print(jnp.linalg.norm(e_fit, ord=2, axis=1))

    a=1.002
    b=2e-5
    c=2.47
    #a=1.002
    #b=1e-4
    #c=0.001

    e_prior = (jnp.pi/4)*( jnp.arctan((w-a)/b)-jnp.arctan((w-a+1)/b) )+c # face it

    #e_prior = jnp.linalg.norm( ( jnp.arctan((w-a)/b)-jnp.arctan((w-a+1)/b) )+4*jnp.pi+c, ord=1)
    #prior = 4*( jnp.arctan((w.astype(jnp.float64).at[0]-a)/b)-jnp.arctan((w.astype(jnp.float64).at[0]-a+1)/b) )+4*jnp.pi+c
    #hcb.id_print(prior)
    #hcb.id_print(w.at[0])

    e_sparse = jnp.linalg.norm(w, ord = 1) # a154blancoiribera
    #reg = jnp.linalg.norm(w, ord = 2)
    #e_reg = reg*reg

    #hcb.id_print(e_sparse)
    #residuals_arr = jnp.append(jnp.concatenate([e_fit.ravel(), prior.ravel()]), e_sparse)

    residuals_arr = jnp.append(jnp.concatenate([b_fit*e_fit.ravel(), b_prior*e_prior]), b_sparse*e_sparse)#0.7*e_reg)

    return residuals_arr


@jax.jit
def energy(w, non_fit_params, cmat, former_w, neutral, bs_arr):
    measured_points, rmat, tvec = unpack(non_fit_params)       

    sparse_weighted = neutral+(bs_arr.T@w.reshape(-1)).T
    e_fit = jnp.linalg.norm(project_points(sparse_weighted, rmat, tvec, cmat) - measured_points, ord=1)

    a=1.002
    b=2e-5
    c=2.47
    #e_prior = jnp.linalg.norm( (jnp.pi/4)*( jnp.arctan((w-a)/b)-jnp.arctan((w-a-1)/b) )+c, ord=2)**2

    e_sparse = jnp.linalg.norm(w, ord = 1)
    #nans = jnp.any(jnp.isnan(residuals))
    #if(nans):
    #jax.debug.print(residuals)
    return e_fit+e_sparse


class ExpressionFitting:
    def __init__(self, camera_matrix, bs_mapper=None, bs_to_ignore = None) -> None:
        self.cam_mat = jax.device_put(jnp.array(camera_matrix, dtype=jnp.float64))
        self.bs_mapper = bs_mapper

        ICT_Model = ICTFaceModel68.from_pkl("./common/ICTFaceModel.pkl", load_blendshapes=True)

        self.right_contour_idx = list(range(0,8))
        self.chin_idx = 8
        self.left_contour_idx = list(range(9,17))

        self.n_blendshapes = ICT_Model.n_blendshapes
        self.neutral_sparse_verts = jax.device_put(jnp.array(ICT_Model.neutral_vertices, dtype=jnp.float64))
        self.bs_names = list(ICT_Model.bs_names)
        
        self.name_to_idx = {name: idx for idx, name in enumerate(self.bs_names) }

        self.sparse_bs_arr = jax.device_put(jnp.array(ICT_Model.get_blendshape_arr(), dtype=jnp.float64))

        removed_idxs = []
        for bs_remove in ['cheekPuff_L', 'cheekPuff_R']:
            bs_idx = self.bs_names.index(bs_remove)
            self.bs_names.remove(bs_remove)
            del self.name_to_idx[bs_remove]
            self.sparse_bs_arr = np.delete(self.sparse_bs_arr, bs_idx, 0)
            removed_idxs.append(bs_idx)
            self.n_blendshapes -= 1

        self.removed_idxs = np.array(removed_idxs)

        self._jaxopt_lm = LevenbergMarquardt(residuals, damping_parameter=1e-06, stop_criterion='grad-l2-norm', 
                                     tol=0.001, xtol=0.001, gtol=0.001, solver='cholesky', 
                                     #geodesic=False, verbose=False, 
                                     jit=True, maxiter=500)#, implicit_diff=True, unroll='auto')
        self.lm = jax.jit(self._jaxopt_lm.run)

        self.scipy_lm = ScipyLeastSquares(fun=residuals, loss='linear', 
                                          options={"ftol": 0.001, "xtol": 0.001, "gtol": 0.001}, 
                                          method='lm', dtype=jnp.float64, jit=True).run

        self.scipy_bounded_lbfgs = ScipyBoundedMinimize(method='L-BFGS-B', dtype=np.float64, 
                                                        jit=True, fun=energy, tol=0.001, maxiter=500).run
        self.former_w = jnp.zeros(self.n_blendshapes)
        self.lower_bounds = jnp.zeros(self.n_blendshapes)
        self.upper_bounds = jnp.ones(self.n_blendshapes)


    def fit(self, measured_points, rvec, tvec, method: str, former_w = None):
        assert method in ['jaxopt_lm', 'scipy_lm', 'l-bfgs-b']
        rmat = cv2.Rodrigues(rvec)[0]
        
        if former_w is not None:
            self.former_w = jnp.array(former_w, dtype=jnp.float64)
        
        w_in = jnp.ones(self.n_blendshapes, dtype=jnp.float64)*0.0001
        
        non_fitting_params = jnp.array(self.pack_params(measured_points, rmat, tvec), dtype=jnp.float64)
        
        if method == 'jaxopt_lm':
            out = self.lm(w_in, non_fitting_params, self.cam_mat, self.former_w, self.neutral_sparse_verts, self.sparse_bs_arr)
        elif method == 'scipy_lm':
            out = self.scipy_lm(w_in, non_fitting_params, self.cam_mat, self.former_w, self.neutral_sparse_verts, self.sparse_bs_arr)
        elif method == 'l-bfgs-b':
            bounds = (self.lower_bounds, self.upper_bounds)
            out = self.scipy_bounded_lbfgs(w_in, bounds, non_fitting_params, self.cam_mat, self.former_w, self.neutral_sparse_verts, self.sparse_bs_arr)

        #w_out = jax.device_get(out.params)
        w_out = out.params

        has_nan = jnp.any(jnp.isnan(w_out))
        if has_nan:
            w_out = w_in
            print("Fitting failed")
        
        n_bs_removed = self.removed_idxs.size
        if n_bs_removed > 0:
            w_out = jnp.insert(w_out, self.removed_idxs, 0.)

        if self.bs_mapper is not None:
            return  jdlp.to_dlpack(self.bs_mapper(w_out), take_ownership=True)
        else:
            return jdlp.to_dlpack(w_out, take_ownership=True)


    def pack_params(self, measured_points, rmat, tvec):
        fitting = np.zeros((68+3+1, 3), dtype=np.float64)
        
        fitting[0:68, 0:2]  = measured_points[:,:]
        fitting[68:68+3, :] = rmat[:,:]
        fitting[68+3]       = tvec.reshape(-1)[:]
        
        return fitting