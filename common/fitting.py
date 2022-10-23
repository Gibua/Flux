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
    transformed = (points@rmat.T + tvec)@cmat.T
    projected = jnp.divide(transformed[:,:2], transformed[:,2:])
    return projected

def ortho_proj(points, rmat, tvec, cmat):
    scale = jnp.divide(cmat[0][0], tvec[2])
    c_x = cmat[0][2]
    c_y = cmat[1][2]
    t_x = c_x + tvec[0]*scale
    t_y = c_y + tvec[1]*scale
    translation = jnp.array([t_x, t_y, 0])
    projected = ((points*scale).dot(rmat.T) + translation)[:,2]
    return projected


#def residuals(w, measured_points, rmat, tvec, cmat, neutral, bs_arr):
@jax.jit
def residuals(w, non_fit_params, cmat, former_w, neutral, bs_arr, b_fit, b_prior, b_sparse):
    measured_points, rmat, tvec = unpack(non_fit_params)

    tvec_flat = tvec.flatten()
    
    #scale = jnp.divide(cmat[0][0], tvec_flat[2])

    sparse_weighted = neutral+(bs_arr.T@w.reshape(-1)).T
    
    #e_fit =(project_points(sparse_weighted, rmat, tvec_flat, cmat)/scale) - (measured_points/scale) # face it
    e_fit = project_points(sparse_weighted, rmat, tvec_flat, cmat) - measured_points # face it

    #hcb.id_print(e_fit)

    a=1.002
    b=2e-5
    #c=2.47
    c=2.47
    d=0.0002

    e_prior = (c/jnp.pi)*( jnp.arctan((w-a)/b)-jnp.arctan((w-a+1)/b) )+c+d # face it

    #e_prior = jnp.linalg.norm( ( jnp.arctan((w-a)/b)-jnp.arctan((w-a+1)/b) )+4*jnp.pi+c, ord=1)
    #prior = 4*( jnp.arctan((w.astype(jnp.float64).at[0]-a)/b)-jnp.arctan((w.astype(jnp.float64).at[0]-a+1)/b) )+4*jnp.pi+c
    #hcb.id_print(prior)
    #hcb.id_print(w.at[0])

    e_sparse =  jnp.array(jnp.linalg.norm(w, ord = 1)).reshape(-1,) # a154blancoiribera
    #e_sparse = jnp.array(jnp.linalg.norm(w, ord = 2)**2).reshape(-1,)
    #e_sparse = reg*reg

    #e_temp = jnp.array(jnp.linalg.norm(former_w-w, ord = 2)**2).reshape(-1,)

    #hcb.id_print(e_sparse)
    #residuals_arr = jnp.append(jnp.concatenate([e_fit.ravel(), prior.ravel()]), e_sparse)

    residuals_arr = jnp.concatenate((b_fit*e_fit.ravel(), b_prior*e_prior, b_sparse*e_sparse))#,0.7*e_reg])

    return residuals_arr


@jax.jit
def energy(w, non_fit_params, cmat, former_w, neutral, bs_arr, b_fit, b_prior, b_sparse):
    measured_points, rmat, tvec = unpack(non_fit_params)       

    sparse_weighted = neutral+(bs_arr.T@w.reshape(-1)).T
    e_fit = jnp.linalg.norm(project_points(sparse_weighted, rmat, tvec, cmat) - measured_points, ord=2)**2

    #a=1.002
    #b=2e-5
    #c=2.47
    #e_prior = jnp.linalg.norm( (jnp.pi/4)*( jnp.arctan((w-a)/b)-jnp.arctan((w-a-1)/b) )+c, ord=2)**2

    e_sparse = jnp.linalg.norm(w, ord = 1)
    e_prior = jnp.linalg.norm(former_w-w, ord = 2)**2
    #nans = jnp.any(jnp.isnan(residuals))
    #if(nans):
    #jax.debug.print(residuals)
    return b_fit*e_fit+b_sparse*e_sparse+b_prior*e_prior


@partial(jax.jit, static_argnames=['n_full'])
def map_to_original(w, n_full, idxs):
    n_fitted = w.size
    w_out = jnp.zeros(n_full, dtype=jnp.float64)
    w_out = w_out.at[idxs].set(w)
    #for i in range(n_fitted):
    #    w_out = w_out.at[idxs[i]].set(w[i])
    return w_out


class ExpressionFitting:
    def __init__(self, camera_matrix, bs_mapper=None, 
                       bs_to_ignore = ['cheekPuff_L', 'cheekPuff_R', 'eyeLookDown_L', 'eyeLookDown_R', 
                                       'eyeLookIn_L', 'eyeLookIn_R', 'eyeLookOut_L', 'eyeLookOut_R', 
                                       'eyeLookUp_L', 'eyeLookUp_R']) -> None:
                       #bs_to_ignore = ['cheekPuff_L', 'cheekPuff_R']) -> None:
        self.cam_mat = jax.device_put(jnp.array(camera_matrix, dtype=jnp.float64))
        
        self.bs_mapper = bs_mapper

        ICT_Model = ICTFaceModel68.from_pkl("./common/ICTFaceModel.pkl", load_blendshapes=True)

        self.right_contour_idx = list(range(0,8))
        self.chin_idx = 8
        self.left_contour_idx = list(range(9,17))

        self.n_bs = ICT_Model.n_blendshapes
        self.neutral_sparse_verts = jax.device_put(jnp.array(ICT_Model.neutral_vertices, dtype=jnp.float64))
        self.bs_names = list(ICT_Model.bs_names)
        
        self.name_to_idx = {name: idx for idx, name in enumerate(self.bs_names)}

        sparse_bs_arr_full = jax.device_put(jnp.array(ICT_Model.get_blendshape_arr(), dtype=jnp.float64))

        fitted_bs_idxs_list = []
        for i, name in enumerate(self.bs_names):
            if name not in bs_to_ignore:
                fitted_bs_idxs_list.append(i)

        self.fitted_bs_idxs = jnp.array(fitted_bs_idxs_list)
        self.n_fitted_bs = self.fitted_bs_idxs.size
        self.sparse_bs_arr = sparse_bs_arr_full[self.fitted_bs_idxs,:,:]
        #removed_idxs = []
        #for bs_remove in bs_to_ignore:
        #    bs_idx = self.bs_names.index(bs_remove)
        #    self.bs_names.remove(bs_remove)
        #    del self.name_to_idx[bs_remove]
        #    self.sparse_bs_arr = np.delete(self.sparse_bs_arr, bs_idx, 0)
        #    removed_idxs.append(bs_idx)
        #    self.n_bs -= 1

        #self.removed_idxs = np.sort(np.array(removed_idxs))

        #idxs_to_insert = []
        #for i, idx in enumerate(self.removed_idxs):
        #    idxs_to_insert.append(idx-i)

        #self.idxs_to_insert = np.array(idxs_to_insert)

        self._jaxopt_lm = LevenbergMarquardt(residuals, damping_parameter=1e-03, stop_criterion='grad-l2-norm', 
                                     tol=0.0001, xtol=0.0001, gtol=0.0001, solver='cholesky', 
                                     #geodesic=True, verbose=False, 
                                     jit=True)#, maxiter=500)#, implicit_diff=True, unroll='auto')
        self.lm = jax.jit(self._jaxopt_lm.run)

        self.scipy_lm = ScipyLeastSquares(fun=residuals, loss='linear',
                                          options={"ftol": 0.001, "xtol": 0.001, "gtol": 0.001, "max_nfev":100}, 
                                          method='lm', dtype=jnp.float64, jit=True).run

        self.scipy_bounded_lbfgs = ScipyBoundedMinimize(method='L-BFGS-B', dtype=jnp.float64, 
                                                        jit=True, fun=energy, tol=0.0001, maxiter=500).run
        self.former_w = jnp.zeros(self.n_fitted_bs)
        self.lower_bounds = jnp.zeros(self.n_fitted_bs)
        self.upper_bounds = jnp.ones(self.n_fitted_bs)

        non_fitting_params = jnp.array(self.pack_params(np.ones((68,2), dtype=np.float64), cv2.Rodrigues(np.array([0,0,0], dtype=np.float64))[0], np.ones(3,dtype=np.float64)), dtype=jnp.float64)
        self.lm(jnp.ones(self.n_fitted_bs, dtype=jnp.float64)*0.0001, non_fitting_params, self.cam_mat, self.former_w, self.neutral_sparse_verts, self.sparse_bs_arr, 1., 1., 1.)


    def fit(self, measured_points, rvec, tvec, b_fit, b_prior, b_sparse, method: str, debug = False):
        assert method in ['jaxopt_lm', 'scipy_lm', 'l-bfgs-b']
        
        rmat = cv2.Rodrigues(rvec)[0]
        #
        w_in = jnp.ones(self.n_fitted_bs, dtype=jnp.float64)*0.0001
        
        non_fitting_params = jnp.array(self.pack_params(measured_points, rmat, tvec), dtype=jnp.float64)
        st = time.perf_counter()
        if method == 'jaxopt_lm':
            out = self.lm(w_in, non_fitting_params, self.cam_mat, self.former_w, self.neutral_sparse_verts, self.sparse_bs_arr, b_fit, b_prior, b_sparse)
        elif method == 'scipy_lm':
            out = self.scipy_lm(w_in, non_fitting_params, self.cam_mat, self.former_w, self.neutral_sparse_verts, self.sparse_bs_arr, b_fit, b_prior, b_sparse)
        elif method == 'l-bfgs-b':
            bounds = (self.lower_bounds, self.upper_bounds)
            out = self.scipy_bounded_lbfgs(w_in, bounds, non_fitting_params, self.cam_mat, self.former_w, self.neutral_sparse_verts, self.sparse_bs_arr, b_fit, b_prior, b_sparse)
        print("=========",time.perf_counter() - st)
        #w_out = jax.device_get(out.params)
        w_out = jnp.clip(out.params, 0, 1)
        #w_out = out.params
        

        has_nan = jnp.any(jnp.isnan(w_out))
        if has_nan:
            w_out = w_in
            print("Fitting failed")
        else:
            self.former_w = w_out.copy()

        #n_bs_removed = self.idxs_to_insert.size
        #if n_bs_removed > 0:
        #    w_out = jnp.insert(w_out, self.idxs_to_insert, 0.)
        if self.n_fitted_bs < self.n_bs:
            #w_full = jnp.zeros(self.n_bs, dtype=jnp.float32)
            #w_full.at[self.fitted_bs_idxs].set(w_out)
            #print(self.fitted_bs_idxs)
            #w_out = w_full
            w_out = map_to_original(w_out, self.n_bs, self.fitted_bs_idxs)

        if debug:
            return self.post_process(w_in, w_out), jax.device_get(w_out)
        else:
            return self.post_process(w_in, w_out)

        
    def post_process(self, w_in, w_out):

        if self.bs_mapper is not None:
            return jdlp.to_dlpack(self.bs_mapper(w_out), take_ownership=True)
        else:
            return jdlp.to_dlpack(w_out.copy(), take_ownership=True)


    def pack_params(self, measured_points, rmat, tvec):
        fitting = np.zeros((68+3+1, 3), dtype=np.float64)
        
        fitting[0:68, 0:2]  = measured_points[:,:]
        fitting[68:68+3, :] = rmat[:,:]
        fitting[68+3]       = tvec.reshape(-1)[:]
        
        return fitting