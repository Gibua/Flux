from common.landmark_mapping import LandmarkMapper
from common.mappings import Datasets

map = LandmarkMapper(Datasets.WFLW, Datasets.IBUG)
print(map.as_list())
