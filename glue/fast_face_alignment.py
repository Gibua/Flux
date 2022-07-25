from utils.face_detection_utils import crop
from fast_face_alignment.utils.utils import get_preds_fromhm
from fast_face_alignment.FastFaceAlignment import FastFAN, generateFastFan


class Predictor(Detector):
	
	def __init__(self):
		model_path = './fast_face_alignment/models/facealignment/FastFAN.pth'
		self.model = generateFastFan(modelPath=model_path, deviceID=0)
		

	def set_input_tensor(self, interpreter, image):
		tensor_index = interpreter.get_input_details()[0]['index']
		print(interpreter.get_input_details())
		interpreter.set_tensor(tensor_index, image)
	
	def pre_process(self, img):
		if len(img) == 0:
			return None
		resized = cv2.resize(cropped, dsize=(int(256), int(256)),
					interpolation=cv2.INTER_LINEAR)
		tensor = torch.from_numpy(resized.transpose(
			(2, 0, 1))).float()
		tensor = tensor.to(self.device)
		tensor.div_(255.0).unsqueeze_(0)
		return tensor
	
	def predict(self, img):
		input_image = self.pre_process(img)
		
		out = self.model.forward(input_image)[-1].cpu()
		
		landmarks = self.interpreter.get_tensor(self.output_details[0]['index'])
		landmarks = expit(landmarks)
		landmarks = landmarks.reshape(-1, 2)
		
		return landmarks
		
	def post_process(self, out, bbox):
		center = torch.FloatTensor(
				[bbox[2] - (bbox[2] - bbox[0]) / 2.0, bbox[3] - (bbox[3] - bbox[1]) / 2.0])
		center[1] = center[1] - (bbox[3] - bbox[1]) * 0.12
		scale = (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / 195
		
		pts, pts_img = get_preds_fromhm(out, centers, scales)
		
		return pts_img.numpy()
