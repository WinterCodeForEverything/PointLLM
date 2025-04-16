import os
import logging
import torch
import math
from typing import List
from util import dist_utils
from tqdm import tqdm
import numpy as np
from datetime import datetime
import open3d as o3d
from whitebox.FGM.GeoA3_args import uniform_loss
from pytorch3d.loss import chamfer_distance

class PointCloudEvaluator:
    def __init__(self, k_nn=4, device='cuda'):
        
        #metrics in HiT_Adv
        self.k_nn = k_nn
        self.knn_dist_metric = dist_utils.KNNDist(k=k_nn).to(device)
        #self.uniform_dist_metric = uniform_loss
        self.curv_std_metric = dist_utils.CurvStdDist(k=k_nn).to(device)
        
    def evaluate(self, ori_pc, adv_pc, ori_normal ):
        
        
        knn_dist = self.knn_dist_metric.forward(pc=adv_pc, weights=None, batch_avg=True)
        uniform_dist = uniform_loss(adv_pc=adv_pc, k=self.k_nn)
        curv_std_dist = self.curv_std_metric.forward(ori_data=ori_pc, adv_data=adv_pc, ori_normal=ori_normal)
        
        
        return {
            'knn_dist': knn_dist,
            'uniform_dist': uniform_dist,
            'curv_std_dist': curv_std_dist
        }
    
    

