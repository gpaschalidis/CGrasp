# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Universiteit van Amsterdam (UvA).
# All rights reserved.
#
# Universiteit van Amsterdam (UvA) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with UvA or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: g.paschalidis@uva.nl
#
import open3d as o3d
import os
import argparse
from cgrasp.tools.cfg_parser import Config
from cgrasp.test.tester import Tester
from cgrasp.test.grasp import Grasp
from cgrasp.tools.cfg_parser import Config


def main():
    parser = argparse.ArgumentParser(description="Generate Grasps")

    parser.add_argument("--obj_path",
                        required=True,   
                        type=str,
                        help="The path to the 3D object Mesh or Pointcloud dataset"
    )
    parser.add_argument("--rhm_path", 
                        required=True,
                        type=str,
                        help="The path to the folder containing MANO_RIHGT model"
    )
    parser.add_argument("--scale", 
                        default=1., 
                        type=float,
                        help="The scaling for the 3D object"
    )
    parser.add_argument("--grasp_type", 
                        default="right", 
                        choices=["left", "right"],
                        help="Scesify the grasp type"
    )
    parser.add_argument("--vis", 
                        action="store_true", 
                        help="Scesify if you want to visualize the generated grasps"
    )
    parser.add_argument("--save_dir", 
                        required=True,
                        default=str, 
                        help="The path to save the generated hand vertices"
    )
    parser.add_argument("--n_samples", 
                        default=10, 
                        type=int,
                        help="number of grasps to generate"
    )
    parser.add_argument("--config_path", 
                        default="cgrasp/pretrained/cgrasp_cfg.yaml",
                        type=str,
                        help="The path to the confguration of the pre trained CGrasp model"
    )
    args = parser.parse_args()
    
    obj_path = args.obj_path
    rhm_path = args.rhm_path
    scale = args.scale
    n_samples = args.n_samples
    cfg_path = args.config_path
    grasp_type = args.grasp_type
    save_dir = args.save_dir
    vis = args.vis
    
    best_rnet = 'cgrasp/pretrained/refinenet.pt'
    bps_dir = 'cgrasp/configs/bps.npz'
    closed_mano_faces_path = "cgrasp/configs/mano_closed_faces.npz"
    config = {
        'bps_dir': bps_dir,
        'rhm_path': rhm_path,
        'best_rnet': best_rnet,
        "save_dir": save_dir
    }
        
    cfg = Config(default_cfg_path=cfg_path, **config)

    cgrasp_tester = Tester(cfg=cfg)

    grasp = Grasp(cgrasp_tester, obj_path, save_dir, closed_mano_faces_path, scale=1)
    grasp.set_input_params(n_samples, grasp_type=grasp_type)
    grasp.save_visualize_grasps(vis=vis)

if __name__ == '__main__':
    main()
