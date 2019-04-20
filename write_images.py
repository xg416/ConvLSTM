import numpy as np
import cv2
import os

#path = r'/home/isat-deep/Documents/isoGD/isoGD/IsoGD_phase_1/IsoGD_phase_1/valid'
path = r'/home/isat-deep/Documents/isoGD/isoGD/IsoGD_phase_2/IsoGD_phase_2/test'
w_path = r'/home/isat-deep/Documents/isoGD/test_images'
for folder in os.listdir(path):
	f_path = os.path.join(path, folder)
	for video in os.listdir(f_path):
		v_path = os.path.join(f_path, video)
		cap = cv2.VideoCapture(v_path)
		if video[0] == 'K':
			write_path = os.path.join(w_path, 'Depth', video)
		elif video[0] == 'M':
			write_path = os.path.join(w_path, 'RGB', video)
		write_path = write_path.split('.')
		os.mkdir(write_path[0])
		i = 0
		while(cap.isOpened()):
			ret, frame = cap.read()
			if(not ret): break
			frame_idx = '%06d' %i
			cv2.imwrite(os.path.join(write_path[0], frame_idx + '.jpg'), frame)
			i += 1
		cap.release()
