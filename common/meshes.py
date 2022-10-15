import numpy as np

from typing import List, Optional
import numpy.typing as npt
from scipy.sparse import csr_matrix

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

    def __init__(self, neutral_vertices: np.ndarray, tri_faces: np.ndarray,
                       blendshapes_arr: Optional[np.ndarray] = None,
                       name_list: Optional[npt.ArrayLike] = None) -> None:
        # blendshapes_arr shape = (n of blendshapes, n of vertices, dimensions (3))

        self.neutral_vertices = neutral_vertices
        self.faces = tri_faces
        self.n_vertices = neutral_vertices.shape[0]

        if blendshapes_arr is not None or name_list is not None:
            assert name_list is not None, "blendshape names not provided"
            assert blendshapes_arr is not None, "array of blendshape vertices not provided"

            self.bs_names = name_list
            self.n_blendshapes = len(name_list)

            assert self.n_blendshapes == blendshapes_arr.shape[0], "number of blendshapes in blendshapes_arr do not match with the provided names (check array shape)"
            assert blendshapes_arr.shape[1] == self.neutral_vertices.shape[0], "number of vertices in the blendshapes does not match neutral mesh"

            self._name_to_idx = {name: idx for idx, name in enumerate(self.bs_names) }

            x = blendshapes_arr.T[0]
            y = blendshapes_arr.T[1]
            z = blendshapes_arr.T[2]

            r_x = ((np.absolute(x) > 1e-5)*x)
            r_y = ((np.absolute(y) > 1e-5)*y)
            r_z = ((np.absolute(z) > 1e-5)*z)

            self.sparse_bs_x = csr_matrix(r_x)
            self.sparse_bs_y = csr_matrix(r_y)
            self.sparse_bs_z = csr_matrix(r_z)

            self.apply_weights      = self._apply_weights
            self.get_blendshape_arr = self._get_blendshape_arr
            self.get_blendshape     = self._get_blendshape
            self.bs_name_from_idx   = self._bs_name_from_idx
            self.bs_idx_from_name   = self._bs_idx_from_name


    def _apply_weights(self, w):
        return self.neutral_vertices + np.array([self.sparse_bs_x@w,
                                                 self.sparse_bs_y@w,
                                                 self.sparse_bs_z@w]).squeeze().T


    def _get_blendshape(self, idx: int):
        return np.array([self.sparse_bs_x[:, idx].toarray(),
                         self.sparse_bs_y[:, idx].toarray(),
                         self.sparse_bs_z[:, idx].toarray()]).T.squeeze()


    def _get_blendshape_arr(self):
        return np.array([self.sparse_bs_x.toarray(),
                         self.sparse_bs_y.toarray(),
                         self.sparse_bs_z.toarray()]).T


    def _bs_name_from_idx(self, idx: int):
        return self.bs_names[idx]


    def _bs_idx_from_name(self, name: str):
        return self._name_to_idx[name]