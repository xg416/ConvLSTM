# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:30:52 2019
reference: https://blog.csdn.net/qq_32799915/article/details/85704240 
@author: Xingguang Zhang
"""
import os
import numpy as np
import cv2

def cal_for_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    i = 0
    flow = []
    while(cap.isOpened()):
        ret, curr = cap.read()
        if(not ret): break
        if i == 0:
            prev = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        else:
            curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            tmp_flow = compute_TVL1(prev, curr)
            flow.append(tmp_flow)
            prev = curr
        i += 1
    return flow

def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow
    
def save_flow(video_flows, flow_path):
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path.format('u'), "{:06d}.jpg".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path.format('v'), "{:06d}.jpg".format(i)),
                    flow[:, :, 1])

def extract_flow(video_path,flow_path):
    flow = cal_for_frames(video_path)
    save_flow(flow, flow_path)
    print('complete:' + flow_path)
    return


if __name__ =='__main__':

	path = r'/home/isat-deep/Documents/isoGD/isoGD/IsoGD_phase_2/IsoGD_phase_2/test'
	w_path = r'/home/isat-deep/Documents/isoGD/test_images/Flow'
	for folder in os.listdir(path):
		f_path = os.path.join(path, folder)
		for video in os.listdir(f_path):
			if video[0] == 'M':
				# create original RGB video path
				v_path = os.path.join(f_path, video)
				# create flow path
				video_idx = video.split('.')
				video_idx = video_idx[0]
				FlowFolder = 'F' + video_idx[1:]
				write_path = os.path.join(w_path, FlowFolder)
				os.mkdir(write_path)
				extract_flow(v_path, write_path)
