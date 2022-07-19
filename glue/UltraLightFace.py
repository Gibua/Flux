from ..UltraLightFace.utils import *
from ..UltraLightFace.TFLiteFaceDetector import UltraLightFaceDetecion
from ..utils.face_detection_utils import square_box

class Predictor:
	
	def __init__(self):	
		self.predictor = UltraLightFaceDetecion("../UltraLightFace/pretrained/version-RFB-320_without_postprocessing.tflite", conf_threshold=0.5)
		self.interpreter.allocate_tensors()
		self.input_details = interpreter.get_input_details()
		self.output_details = interpreter.get_output_details()
		
	def predict_bbox(img):
		boxes, scores = self.predictor.inference(img)
		
		boxes = np.sort(boxes, axis=0)
		result = boxes[0]
		
		result[1] = int(result[1]*(1+0.1))
		result[3] = int(result[3]*(1+0.1))
		
		bbox = square_box(result)
		
		return bbox
		
	
