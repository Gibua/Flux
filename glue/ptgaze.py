import cv2
import numpy as np
import sys, os
import copy

import yaml
import tempfile
from omegaconf import DictConfig

# from modules.ptgaze.utils import *


class GazeEstimator():
	
	def __init__(self, camera_matrix, width, height):
		self.config = _load_mode_config()
		self._generate_dummy_camera_params()
		
		self.gaze_estimator = GazeEstimator(config)

		self.cap = self._create_capture()
		self.output_dir = self._create_output_dir()
		self.writer = self._create_video_writer()

		self.stop = False
		self.show_bbox = self.config.demo.show_bbox
		self.show_head_pose = self.config.demo.show_head_pose
		self.show_landmarks = self.config.demo.show_landmarks
		self.show_normalized_image = self.config.demo.show_normalized_image
		self.show_template_model = self.config.demo.show_template_model
		
	
	def pre_process(self, img):
		pass
	
	
	def extract_eye_image_patches(img, landmarks, bbox, eye_corner_indices):
		re_c, le_c, _, _ = get_eye_image_from_landmarks(img, landmarks, bbox, eye_corner_indices)
		return re_c, le_c
	
	
	def rvec_to_theta_phi(self, rvec):
		_rotation_matrix, _ = cv2.Rodrigues(rvec)
		_rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
		_m = np.zeros((4, 4))
		_m[:3, :3] = _rotation_matrix
		_m[3, 3] = 1
		# Go from camera space to ROS space
		_camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
						  [-1.0, 0.0, 0.0, 0.0],
						  [0.0, -1.0, 0.0, 0.0],
						  [0.0, 0.0, 0.0, 1.0]]
		roll_pitch_yaw = list(euler_from_matrix(np.dot(_camera_to_ros, _m)))
		roll_pitch_yaw = limit_yaw(roll_pitch_yaw)
		
		phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)
		
		return phi, theta
	
	
	def _generate_camera_params(self, config: DictConfig) -> None:
		out_file = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
		dic = {
			'image_width': self.width,
			'image_height': self.height,
			'camera_matrix': {
				'rows': 3,
				'cols': 3,
				'data': self.camera_matrix
			},
			'distortion_coefficients': {
				'rows': 1,
				'cols': 5,
				'data': [0., 0., 0., 0., 0.]
			}
		}
		with open(out_file.name, 'w') as f:
			yaml.safe_dump(dic, f)
		config.gaze_estimator.camera_params = out_file.name
	
	
	def _check_path(config: DictConfig, key: str) -> None:
		path = operator.attrgetter(key)(config)
		if not os.path.exists(path):
			raise FileNotFoundError(f'config.{key}: {path} not found.')
		if not  os.path.isfile(path):
			raise ValueError(f'config.{key}: {path} is not a file.')
	
	
	def _check_path_all(config: DictConfig) -> None:
		_check_path(config, 'gaze_estimator.checkpoint')
		_check_path(config, 'gaze_estimator.camera_params')
		_check_path(config, 'gaze_estimator.normalized_camera_params')
	
	
	def download_mpiigaze_model() -> pathlib.Path:
		output_dir = pathlib.Path('~/.ptgaze/models/').expanduser()
		output_dir.mkdir(exist_ok=True, parents=True)
		output_path = output_dir / 'mpiigaze_resnet_preact.pth'
		if not output_path.exists():
			logger.debug('Download the pretrained model')
			torch.hub.download_url_to_file(
				'https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.1.0/mpiigaze_resnet_preact.pth',
				output_path.as_posix())
		else:
			logger.debug(f'The pretrained model {output_path} already exists.')
		return output_path


	def download_mpiifacegaze_model() -> pathlib.Path:
		output_dir = pathlib.Path('~/.ptgaze/models/').expanduser()
		output_dir.mkdir(exist_ok=True, parents=True)
		output_path = output_dir / 'mpiifacegaze_resnet_simple.pth'
		if not output_path.exists():
			logger.debug('Download the pretrained model')
			torch.hub.download_url_to_file(
				'https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.1.0/mpiifacegaze_resnet_simple.pth',
				output_path.as_posix())
		else:
			logger.debug(f'The pretrained model {output_path} already exists.')
		return output_path


	def download_ethxgaze_model() -> pathlib.Path:
		output_dir = pathlib.Path('~/.ptgaze/models/').expanduser()
		output_dir.mkdir(exist_ok=True, parents=True)
		output_path = output_dir / 'eth-xgaze_resnet18.pth'
		if not output_path.exists():
			logger.debug('Download the pretrained model')
			torch.hub.download_url_to_file(
				'https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.2.2/eth-xgaze_resnet18.pth',
				output_path.as_posix())
		else:
			logger.debug(f'The pretrained model {output_path} already exists.')
		return output_path
	
	
	def _generate_config(self) -> DictConfig:
		package_root = './modules/ptgaze/'
		
		path = os.path.join(package_root, 'data/configs/mpiigaze.yaml')
		# path = os.path.join(package_root, 'data/configs/mpiifacegaze.yaml')
		# path = os.path.join(package_root, 'data/configs/eth-xgaze.yaml')
		
		config = OmegaConf.load(path)
		config.PACKAGE_ROOT = package_root
		config.device = self.device
		_generate_camera_params(config)
		if config.device == 'cuda' and not torch.cuda.is_available():
			config.device = 'cpu'
			#warnings.warn('Run on CPU because CUDA is not available.')
			
		OmegaConf.set_readonly(config, True)
		_check_path_all(config)
	
		return config
