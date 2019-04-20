# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:30:52 2019

@author: 10659
"""
import os
import numpy as np
import cv2
from glob import glob

_IMAGE_SIZE = 256

def cal_for_frames(video_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow

def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create()
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

    video_paths="/home/xueqian/bishe/extrat_feature/output"
    flow_paths="/home/xueqian/bishe/extrat_feature/flow"
    video_lengths = 109 

    extract_flow(video_paths, flow_paths)
--------------------- 
作者：qq_32799915 
来源：CSDN 
原文：https://blog.csdn.net/qq_32799915/article/details/85704240 
版权声明：本文为博主原创文章，转载请附上博文链接！
        