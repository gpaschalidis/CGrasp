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

import os
import numpy as np
import torch

from cgrasp.tools.utils import makepath, makelogger
from cgrasp.models.models import CGrasp, RefineNet



class Tester:

    def __init__(self, cfg):

        self.dtype = torch.float32

        makepath(cfg.save_dir, isfile=False)
        logger = makelogger(makepath(os.path.join(cfg.save_dir, 'V00.log'), isfile=True)).info
        self.logger = logger
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cgrasp = CGrasp(latentD=cfg.latentD)

        self.refine_net = RefineNet().to(self.device)

        self.cfg = cfg
        self.cgrasp.cfg = cfg

        if cfg.cgrasp_model_path is not None:
            self._get_cgrasp_model().load_state_dict(torch.load(cfg.cgrasp_model_path, map_location=self.device), strict=False)
            logger('Restored CGrasp model from %s' % cfg.cgrasp_model_path)
        if cfg.best_rnet is not None:
            self._get_rnet_model().load_state_dict(torch.load(cfg.best_rnet, map_location=self.device), strict=False)
            logger('Restored RefineNet model from %s' % cfg.best_rnet)

        self.bps = torch.from_numpy(np.load(cfg.bps_dir)['basis']).to(self.dtype)

    def _get_cgrasp_model(self):
        return self.cgrasp

    def _get_rnet_model(self):
        return self.refine_net
