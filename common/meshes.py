import numpy as np

from typing import List


#class Mesh:
#
#    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
#        self.vertices = vertices
#        self.faces = faces


#class DeltaBlendshape:
#
#    def __init__(self, delta_vertices: np.ndarray, name: str) -> None:
#        self.name = name
#        self.vertices = delta_vertices
#
#    @classmethod
#    def from_meshes(cls, neutral_mesh: Mesh, blendshape_mesh: Mesh, name: str):
#        delta = blendshape_mesh.vertices - neutral_mesh.vertices
#        return cls(delta, name)

class Model:

    def __init__(self, neutral_vertices: np.ndarray, tri_faces: np.ndarray, blendshapes_arr: np.ndarray, name_list) -> None:
        self.neutral_vertices = neutral_vertices
        self.faces = tri_faces

        self.n_blendshapes = len(name_list)
        self.n_vertices = neutral_vertices.shape[0]

        self.d_blendshape_mat = blendshapes_arr

        #x = d_bss[:,0,:]
        #y = d_bss[:,1,:]
        #z = d_bss[:,2,:]

        #r_x = ((np.absolute(x) > 1e-5)*x)
        #r_y = ((np.absolute(y) > 1e-5)*y)
        #r_z = ((np.absolute(z) > 1e-5)*z)

        #s_x = csr_matrix(r_x)
        #s_y = csr_matrix(r_y)
        #s_z = csr_matrix(r_z)
        
        

    @classmethod
    def from_dict(cls, neutral_vertices: np.ndarray, tri_faces: np.ndarray, blendshape_dict: dict):
        blendshape_keys = list(blendshape_dict.keys())

        bs_names = {}
        for i, name in enumerate(blendshape_keys):
            bs_names[name] = i
        d_blendshape_mat = np.moveaxis( np.array(list(blendshape_dict['deltas'].values())), [0, 1, 2], [2, 0, 1] )

        return cls(neutral_vertices, tri_faces, d_blendshape_mat, bs_names)

    #def apply_weights(self, w):
    #    return np.array([self.sparse_x@w.flatten(),
    #                     self.sparse_y@w.flatten(),
    #                     self.sparse_z@w.flatten()]).T