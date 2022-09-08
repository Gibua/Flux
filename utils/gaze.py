import cv2

def align_face_to_eyes(face_img, right_eye_center, left_eye_center, face_width=None, face_height=None):
	# original from https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
	# modified from https://github.com/Tobias-Fischer/rt_gene/blob/master/rt_gene/src/rt_gene/tracker_generic.py
	desired_left_eye = (0.35, 0.35)
	desired_face_width = face_width if face_width is not None else face_img.shape[1]
	desired_face_height = face_height if face_height is not None else face_img.shape[0]
	# compute the angle between the eye centroids
	d_y = right_eye_center[1] - left_eye_center[1]
	d_x = right_eye_center[0] - left_eye_center[0]
	angle = np.degrees(np.arctan2(d_y, d_x)) - 180

	# compute the desired right eye x-coordinate based on the
	# desired x-coordinate of the left eye
	desired_right_eye_x = 1.0 - desired_left_eye[0]

	# determine the scale of the new resulting image by taking
	# the ratio of the distance between eyes in the *current*
	# image to the ratio of distance between eyes in the
	# *desired* image
	dist = np.sqrt((d_x ** 2) + (d_y ** 2))
	desired_dist = (desired_right_eye_x - desired_left_eye[0])
	desired_dist *= desired_face_width
	scale = desired_dist / dist

	# compute center (x, y)-coordinates (i.e., the median point)
	# between the two eyes in the input image
	eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
				   (left_eye_center[1] + right_eye_center[1]) // 2)

	# grab the rotation matrix for rotating and scaling the face
	rmat = cv2.getRotationMatrix2D(eyes_center, angle, scale)

	# update the translation component of the matrix
	t_x = desired_face_width * 0.5
	t_y = desired_face_height * desired_left_eye[1]
	rmat[0, 2] += (t_x - eyes_center[0])
	rmat[1, 2] += (t_y - eyes_center[1])

	# apply the affine transformation
	(w, h) = (desired_face_width, desired_face_height)
	aligned_face = cv2.warpAffine(face_img, rmat, (w, h), flags=cv2.INTER_NEAREST)
	return aligned_face, rmat
