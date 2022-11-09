from functools import partial
import numpy as np
import jax.numpy as jnp
import sys
import torch

def generate_w_mapper(src_name_list, tgt_name_list, use_jax = False):
    brows_src_idx = None
    cheeks_src_idx = None

    brows_in = "browInnerUp" in tgt_name_list
    cheeks_in = "cheekPuff" in tgt_name_list

    src_name_to_idx = {name: idx for idx, name in enumerate(src_name_list)}

    if brows_in:
        brows_src_idx = [src_name_to_idx["browInnerUp_R"], src_name_to_idx["browInnerUp_L"]]

    if cheeks_in:
        cheeks_src_idx = [src_name_to_idx["cheekPuff_R"], src_name_to_idx["cheekPuff_L"]]

    corr_list = []
    for name in tgt_name_list:
        name_to_check = name[:]
        if name in ["browInnerUp", "cheekPuff"]:
            name_to_check += '_R'
        if name_to_check in list(src_name_list):
            corr_list.append(src_name_to_idx[name_to_check])
        else:
            continue
        
    if not use_jax:
        return partial(_map_w, correspondences_list = np.array(corr_list), 
                       brows_idxs = brows_src_idx, 
                       cheeks_idxs = cheeks_src_idx)
    elif use_jax:
        return partial(_jax_map_w, correspondences_list = np.array(corr_list), 
                       brows_idxs = brows_src_idx, 
                       cheeks_idxs = cheeks_src_idx)

def _map_w(w, correspondences_list, brows_idxs = None, cheeks_idxs = None):
    brows_in  = brows_idxs is not None
    cheeks_in = brows_idxs is not None

    w_in = w.flatten()
    #w_in = torch.arange(w.cpu().numpy().size).cuda()

    if brows_in:
        w_in[brows_idxs[0]] = max(w_in[brows_idxs[0]], w_in[brows_idxs[1]])
    if cheeks_in:
        w_in[cheeks_idxs[0]] = max(w_in[cheeks_idxs[0]], w_in[cheeks_idxs[1]])

    w_out = w_in[correspondences_list]

    return w_out


def _jax_map_w(w, correspondences_list, brows_idxs = None, cheeks_idxs = None):
    brows_in  = brows_idxs is not None
    cheeks_in = brows_idxs is not None

    w_in = w.flatten()

    if brows_in:
        w_in = w_in.at[brows_idxs[0]].set(max(w_in[brows_idxs[0]], w_in[brows_idxs[1]]))
    if cheeks_in:
        w_in = w_in.at[cheeks_idxs[0]].set(max(w_in[cheeks_idxs[0]], w_in[cheeks_idxs[1]]))

    w_out = w_in[correspondences_list]

    return w_out