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

import sys
sys.path.append('.')
sys.path.append('..')
import os
import argparse
from cgrasp.tools.cfg_parser import Config
from cgrasp.train.trainer import Trainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CGrasp-Training')
    
    parser.add_argument('--config_path', 
                        required=True,
                        type=str,
                        help='The path to the config file, containing the training configuration details.'
    )
    parser.add_argument('--data_path', 
                        required=True,
                        type=str,
                        help='The path to the folder that contains GrabNet data'
    )
    parser.add_argument('--rhm_path', 
                        required=True,
                        type=str,
                        help='The path to the folder containing MANO_RIHGT model'
    )
    parser.add_argument('--save_dir', 
                        reruired=True,
                        type=str,
                        help='The path to save the results'
    )
    parser.add_argument('--log',
                        default=False,
                        type=lambda arg: arg.lower() in ['true', '1']
    )


    args = parser.parse_args()

    save_dir = args.save_dir
    data_path = args.data_path
    rhm_path = args.rhm_path
    cfg_path = args.config_path    

    cwd = os.getcwd()

    cfg = {
        'dataset_dir': data_path,
        'rhm_path': rhm_path,
        'save_dir': save_dir,
        'log': args.log
    }

    cfg = Config(default_cfg_path=cfg_path, **cfg)
    cgrasp_trainer = Trainer(cfg=cfg)
    
    cgrasp_trainer.fit()

    cfg = cgrasp_trainer.cfg
    cfg.write_cfg(os.path.join(save_dir, 'TR%02d_%s' % (cfg.try_num, os.path.basename(cfg_path))))
