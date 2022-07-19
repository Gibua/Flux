from abc import ABC, abstractmethod

class Detector(ABC):

	@abstractmethod
	def predict(img):
		pass
		
	@abstractmethod
	def pre_process(img):
		pass
	
	@abstractmethod
	def post_process(landmarks, bbox):
		pass


