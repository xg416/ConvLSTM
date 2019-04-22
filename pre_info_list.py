import numpy as np
import cv2
import os

images_path = r'/home/isat-deep/Documents/isoGD/valid_images/Depth'
info_path = r'/home/isat-deep/Documents/isoGD/isoGD/IsoGD_phase_2/IsoGD_phase_2/valid_list_with_label.txt'
#path = r'/home/isat-deep/Documents/isoGD/isoGD/IsoGD_phase_2/IsoGD_phase_2/test'
w_path = r'/home/isat-deep/Documents/isoGD/valid_depth_list.txt'

v_list = os.listdir(images_path)
v_list.sort()
w_list = []
f = open(info_path, 'r')
info_list = f.readlines()
f.close()

assert len(v_list) == len(info_list)

for video, line in zip(v_list, info_list):
    class_idx = str(int(line.split()[2])-1)
    frame_num = len(os.listdir(os.path.join(images_path, video)))
    w_list.append(os.path.join(images_path, video) + ' ' + str(frame_num)+' '+class_idx + '\n')


fo = open(w_path, 'w')

for line in w_list:
    fo.write(line)
fo.close()