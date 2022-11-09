from common.mappings import *

import numpy as np

class LandmarkMapper:

    def __init__(self, source: Datasets, destination: Datasets):
        if source == destination:
            raise ValueError("source and destination can't be the same")
        if source == Datasets.WFLW:
            if destination == Datasets.IBUG:
                map = WFLWToIBUG()
                self._mapping = map.mapping
                self.src_indices = map.wflw_indices
                self.dest_indices = map.ibug_indices
        elif source == Datasets.IBUG:
            if destination == Datasets.WFLW:
                map = WFLWToIBUG()
                self._mapping = self._invert_mapping(map)
                self.src_indices = map.wflw_indices
                self.dest_indices = map.ibug_indices
        else:
            raise ValueError("source not supported")

        self._inverted_dict_mapping = self._invert_mapping(self._mapping)
        self._list = list(self._mapping)
        self._tuples = tuple(self._mapping.items())
        self._np_array = np.array(self.as_list())


    def as_dict(self):
        return self._mapping

    def as_list(self):
        return self._list

    def as_array(self):
        return self._np_array

    def as_tuples(self):
        return self._tuples

    def inverted_map(self):
        return self._inverted_dict_mapping
    
    def _invert_mapping(self, mapping: dict):
        inverted_dict_mapping = {v: k for k, v in mapping.items()}
        return inverted_dict_mapping

    def map_landmarks(self, src_landmarks):
        return src_landmarks[self.as_list()]
