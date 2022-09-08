import numpy as np
from collections import OrderedDict
import enum

class Datasets(enum.Enum):
    IBUG = enum.auto()
    WFLW = enum.auto()


class WFLWToIBUG:

    def __init__(self):
        self.mapping      = self._generate_mapping()
        self.ibug_indices = self._generate_ibug_idx()
        self.wflw_indices = self._generate_wflw_idx()

    def _generate_ibug_idx(self):
        IBUG_68_PTS_MODEL_IDX = {
            "jaw": list(range(0, 17)),
            "left_eyebrow": list(range(17, 22)),
            "right_eyebrow": list(range(22, 27)),
            "nose": list(range(27, 36)),
            "left_eye": list(range(36, 42)),
            "right_eye": list(range(42, 48)),
            "left_eye_poly": list(range(36, 42)),
            "right_eye_poly": list(range(42, 48)),
            "mouth": list(range(48, 68)),
            "eyes": list(range(36, 42)) + list(range(42, 48)),
            "eyebrows": list(range(17, 22)) + list(range(22, 27)),
            "eyes_and_eyebrows": list(range(17, 22)) + list(range(22, 27)) + list(range(36, 42)) + list(range(42, 48)),
        }
        return IBUG_68_PTS_MODEL_IDX

    def _generate_wflw_idx(self):
        WFLW_98_PTS_MODEL_IDX = {
            "jaw": list(range(0, 33)),
            "left_eyebrow": list(range(33, 42)),
            "right_eyebrow": list(range(42, 51)),
            "nose": list(range(51, 60)),
            "left_eye": list(range(60, 68)) + [96],
            "right_eye": list(range(68, 76)) + [97],
            "left_eye_poly": list(range(60, 68)),
            "right_eye_poly": list(range(68, 76)),
            "mouth": list(range(76, 96)),
            "eyes": list(range(60, 68)) + [96] + list(range(68, 76)) + [97],
            "eyebrows": list(range(33, 42)) + list(range(42, 51)),
            "eyes_and_eyebrows": list(range(33, 42)) + list(range(42, 51)) + list(range(60, 68)) + [96] + list(
                range(68, 76)) + [97],
        }
        return WFLW_98_PTS_MODEL_IDX

    def _generate_mapping(self):
        IBUG_68_TO_WFLW_98_IDX_MAPPING = OrderedDict()
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(0, 16+1), range(0, 32+2, 2))))  # jaw | 17 pts
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update(
            dict(zip(range(17, 21+1), range(33, 37+1))))  # left upper eyebrow points | 5 pts
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update(
            dict(zip(range(22, 26+1), range(42, 46+1))))  # right upper eyebrow points | 5 pts
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(27, 35+1), range(51, 59+1))))  # nose points | 9 pts
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update({36: 60})  # left eye points | 6 pts
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update({37: 61})
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update({38: 63})
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update({39: 64})
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update({40: 65})
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update({41: 67})
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update({42: 68})  # right eye | 6 pts
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update({43: 69})
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update({44: 71})
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update({45: 72})
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update({46: 73})
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update({47: 75})
        IBUG_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(48, 67+1), range(76, 95+1))))  # mouth points | 20 pts

        WFLW_98_TO_IBUG_68_IDX_MAPPING = {v: k for k, v in IBUG_68_TO_WFLW_98_IDX_MAPPING.items()}

        return WFLW_98_TO_IBUG_68_IDX_MAPPING
