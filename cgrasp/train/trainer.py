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
import shutil
import sys
sys.path.append('.')
sys.path.append('..')
import json
import numpy as np
import torch
import mano
import chamfer_distance as chd
from tqdm import tqdm
from datetime import datetime

from cgrasp.tools.utils import makepath, makelogger
from cgrasp.tools.train_tools import EarlyStopping, point2point_signed
from cgrasp.models.models import CGrasp
from cgrasp.data.dataloader import LoadData

from torch import nn, optim
from torch.utils.data import DataLoader

from pytorch3d.structures import Meshes
from tensorboardX import SummaryWriter


class Trainer:

    def __init__(self,cfg, inference=False):

        self.dtype = torch.float32

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        starttime = datetime.now().replace(microsecond=0)
        makepath(cfg.save_dir, isfile=False)
        logger = makelogger(makepath(os.path.join(cfg.save_dir, '%s.log' % (cfg.expr_ID)), isfile=True)).info
        self.logger = logger

        summary_logdir = os.path.join(cfg.save_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        logger('[%s] - Started training CGrasp, experiment code %s' % (cfg.expr_ID, starttime))
        logger('tensorboard --logdir=%s' % summary_logdir)
        logger('Torch Version: %s\n' % torch.__version__)
        logger('Base dataset_dir is %s' % cfg.dataset_dir)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda:%d" % cfg.cuda_id if torch.cuda.is_available() else "cpu")

        gpu_brand = torch.cuda.get_device_name(cfg.cuda_id) if use_cuda else None
        gpu_count = 1
        if use_cuda:
            logger('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))

        self.data_info = {}
        self.load_data(cfg, inference)

        with torch.no_grad():
            self.rhm_train = mano.load(model_path=cfg.rhm_path,
                                       model_type='mano',
                                       num_pca_comps=45,
                                       batch_size=cfg.batch_size,
                                       flat_hand_mean=True).to(self.device)
            
            rhm_train_rn = mano.load(model_path=cfg.rhm_path,
                                       model_type='mano',
                                       num_pca_comps=45,
                                       batch_size=cfg.batch_size // gpu_count,
                                       flat_hand_mean=True).to(self.device)
         
        self.cgrasp = CGrasp(latentD=cfg.latentD).to(self.device)
        
        self.LossL1 = torch.nn.L1Loss(reduction='mean')
        self.LossL2 = torch.nn.MSELoss(reduction='mean')
        self.chamfer_dist = chd.ChamferDistance().to(self.device)


        vars_cgrasp = [var[1] for var in self.cgrasp.named_parameters()]
        
        cgrasp_n_params = sum(p.numel() for p in vars_cgrasp if p.requires_grad)
        logger('Total Trainable Parameters for CGrasp is %2.2f M.' % ((cgrasp_n_params) * 1e-6))

        self.optimizer_cgrasp = optim.Adam(vars_cgrasp, lr=cfg.base_lr, weight_decay=cfg.reg_coef)

        self.best_loss_cgrasp = np.inf

        self.try_num = cfg.try_num
        print('try_num', self.try_num)
        self.epochs_completed = 0
        self.cfg = cfg
        self.cgrasp.cfg = cfg
        if cfg.cgrasp_model_path is not None:
            self._get_cgrasp_model().load_state_dict(torch.load(cfg.cgrasp_model_path, map_location=self.device), strict=False)
            logger('Restored CGrasp model from %s' % cfg.cgrasp_model_path)
        # weights for contact, penetration and distance losses
        self.vpe  = torch.from_numpy(np.load(cfg.vpe_path)).to(self.device).to(torch.long)
        rh_f = torch.from_numpy(self.rhm_train.faces.astype(np.int32)).view(1, -1, 3)
        self.rh_f = rh_f.repeat(self.cfg.batch_size,1,1).to(self.device).to(torch.long)

        v_weights = torch.from_numpy(np.load(cfg.c_weights_path)).to(torch.float32).to(self.device)
        v_weights2 = torch.pow(v_weights, 1.0 / 2.5)

        self.v_weights = v_weights
        self.v_weights2 = v_weights2

        self.w_dist = torch.ones([self.cfg.batch_size,self.n_obj_verts]).to(self.device)
        self.contact_v = v_weights > 0.8

        self.rh_ids_sampled = [1, 4, 8, 13, 22, 24, 35, 45, 47, 63, 79, 82, 85, 89, 90, 97, 102, 104, 122, 144, 156, 158, 161, 171, 173, 177, 179, 182, 185, 188, 190, 212, 216, 220, 223, 231, 246, 248, 256, 264, 267, 268, 275, 283, 298, 300, 314, 333, 342, 346, 350, 357, 361, 367, 379, 381, 388, 393, 407, 426, 431, 444, 448, 453, 462, 470, 489, 491, 495, 498, 500, 508, 516, 537, 546, 555, 559, 564, 573, 589, 607, 609, 613, 618, 625, 658, 669, 672, 676, 690, 706, 710, 712, 724, 733, 744, 749, 755, 763]

    def load_data(self,cfg, inference):

        kwargs = {'num_workers': cfg.n_workers,
                  'batch_size':cfg.batch_size,
                  'shuffle':True,
                  'drop_last':True
                  }
        ds_name = 'test'
        self.data_info[ds_name] = {}
        ds_test = LoadData(dataset_dir=cfg.dataset_dir, ds_name=ds_name)
        self.data_info[ds_name]['frame_names'] = ds_test.frame_names
        self.data_info[ds_name]['frame_sbjs'] = ds_test.frame_sbjs
        self.ds_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        if not inference:
            ds_name = 'train'
            self.data_info[ds_name] = {}
            ds_train = LoadData(dataset_dir=cfg.dataset_dir, ds_name=ds_name)
            self.data_info[ds_name]['frame_names'] = ds_train.frame_names
            self.data_info[ds_name]['frame_sbjs'] = ds_train.frame_sbjs
            self.data_info['hand_vtmp'] = ds_train.sbj_vtemp
            self.data_info['hand_betas'] = ds_train.sbj_betas
            self.ds_train = DataLoader(ds_train, **kwargs)

            ds_name = 'val'
            self.data_info[ds_name] = {}
            ds_val = LoadData(dataset_dir=cfg.dataset_dir, ds_name=ds_name)
            self.data_info[ds_name]['frame_names'] = ds_val.frame_names
            self.data_info[ds_name]['frame_sbjs'] = ds_val.frame_sbjs
            self.ds_val = DataLoader(ds_val, **kwargs)

            self.logger('Dataset Train, Vald, Test size respectively: %.2f M, %.2f K, %.2f K' %
                   (len(self.ds_train.dataset) * 1e-6, len(self.ds_val.dataset) * 1e-3, len(self.ds_test.dataset) * 1e-3))

        self.bps = ds_test.bps
        self.n_obj_verts = ds_test[0]['verts_object'].shape[0]

    def edges_for(self, x, vpe):
        return (x[:, vpe[:, 0]] - x[:, vpe[:, 1]])

    def verts2obj(self, x, y):
        _, _, xidx_near, _ = self.chamfer_dist(x, y)
        xidx_near_expanded = xidx_near.view(x.shape[0], x.shape[1], 1).expand(x.shape[0], x.shape[1], x.shape[2]).to(torch.long)
        x_near = y.gather(1, xidx_near_expanded)
        
        return x - x_near
        
    def _get_cgrasp_model(self):
        return self.cgrasp

    def save_cgrasp(self):
        torch.save(self.cgrasp.state_dict(), self.cfg.cgrasp_model_path)

    def train(self):

        self.cgrasp.train()

        save_every_it = len(self.ds_train) / self.cfg.log_every_epoch

        train_loss_dict_cgrasp = {}
        torch.autograd.set_detect_anomaly(True)

        for it, dorig in tqdm(enumerate(self.ds_train)):

            dorig = {k: dorig[k].to(self.device) for k in dorig.keys()}
            
            dorig['verts2obj'] = self.verts2obj(dorig['verts_rhand'], dorig['verts_object'])[:, self.rh_ids_sampled]

            #if self.cfg.fix_dir:
            dir_wr = dorig['verts_rhand'][:, 220] - dorig['verts_rhand'][:, 216]
            dir_wr = dir_wr / (((dir_wr**2).sum(1)).sqrt()[..., None] + 1e-8)
            dorig['dir_hand'] = dir_wr

            self.optimizer_cgrasp.zero_grad()

            drec_cgrasp = self.cgrasp(**dorig)
            loss_total_cgrasp, cur_loss_dict_cgrasp = self.loss_cgrasp(dorig, drec_cgrasp)

            if self.cfg.log:
                wandb.log({"loss cgrasp": loss_total_cgrasp})

            loss_total_cgrasp.backward()
            self.optimizer_cgrasp.step()

            train_loss_dict_cgrasp = {k: train_loss_dict_cgrasp.get(k, 0.0) + v.item() for k, v in cur_loss_dict_cgrasp.items()}
            if it % (save_every_it + 1) == 0:
                cur_train_loss_dict_cgrasp = {k: v / (it + 1) for k, v in train_loss_dict_cgrasp.items()}
                train_msg = self.create_loss_message(cur_train_loss_dict_cgrasp,
                                                    expr_ID=self.cfg.expr_ID,
                                                    epoch_num=self.epochs_completed,
                                                    model_name='CGrasp',
                                                    it=it,
                                                    try_num=self.try_num,
                                                    mode='train')

                self.logger(train_msg)


        train_loss_dict_cgrasp = {k: v / len(self.ds_train) for k, v in train_loss_dict_cgrasp.items()}

        return train_loss_dict_cgrasp

    def evaluate(self, ds_name='val'):
        self.cgrasp.eval()

        eval_loss_dict_cgrasp = {}

        data = self.ds_val if ds_name == 'val' else self.ds_test

        with torch.no_grad():
            for dorig in data:

                dorig = {k: dorig[k].to(self.device) for k in dorig.keys()}
                
                dorig['verts2obj'] = self.verts2obj(dorig['verts_rhand'], dorig['verts_object'])[:, self.rh_ids_sampled]

                dir_wr = dorig['verts_rhand'][:, 220] - dorig['verts_rhand'][:, 216]
                dir_wr = dir_wr / (((dir_wr**2).sum(1)).sqrt()[..., None] + 1e-8)
                dorig['dir_hand'] = dir_wr

                drec_cgrasp = self.cgrasp(**dorig)
                loss_total_cgrasp, cur_loss_dict_cgrasp = self.loss_cgrasp(dorig, drec_cgrasp)

                eval_loss_dict_cgrasp = {k: eval_loss_dict_cgrasp.get(k, 0.0) + v.item() for k, v in cur_loss_dict_cgrasp.items()}

            eval_loss_dict_cgrasp = {k: v / len(data) for k, v in eval_loss_dict_cgrasp.items()}

        return eval_loss_dict_cgrasp

    def loss_cgrasp(self, dorig, drec, ds_name='train'):

        device = dorig['verts_rhand'].device
        dtype = dorig['verts_rhand'].dtype

        q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])

        out_put = self.rhm_train(**drec)
        verts_rhand = out_put.vertices
        rh_mesh_normals = Meshes(verts=verts_rhand, faces=self.rh_f).to(self.device).verts_normals_packed().view(-1, 778, 3)
        rh_mesh_gt_normals = Meshes(verts=dorig['verts_rhand'], faces=self.rh_f).to(self.device).verts_normals_packed().view(-1, 778, 3)
        o2h_signed, h2o, _ = point2point_signed(verts_rhand, dorig['verts_object'], rh_mesh_normals)
        o2h_signed_gt, h2o_gt, o2h_idx = point2point_signed(dorig['verts_rhand'], dorig['verts_object'], rh_mesh_gt_normals)

        # addaptive weight for penetration and contact verts
        w_dist = (o2h_signed_gt < 0.01) * (o2h_signed_gt > -0.005)
        w_dist_neg = o2h_signed < 0.
        w = self.w_dist.clone()
        w[~w_dist] = .1 # less weight for far away vertices
        w[w_dist_neg] = 1.5 # more weight for penetration
        ######### dist loss
        loss_dist_h = 35 * (1. - self.cfg.kl_coef) * torch.mean(torch.einsum('ij,j->ij', torch.abs(h2o.abs() - h2o_gt.abs()), self.v_weights2))
        loss_dist_o = 30 * (1. - self.cfg.kl_coef) * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2h_signed - o2h_signed_gt), w))
        ########## verts loss
        loss_mesh_rec_w = 35 * (1. - self.cfg.kl_coef) * torch.mean(torch.einsum('ijk,j->ijk', torch.abs((dorig['verts_rhand'] - verts_rhand)), self.v_weights))
        ########## edge loss
        loss_edge = 30 * (1. - self.cfg.kl_coef) * self.LossL1(self.edges_for(verts_rhand, self.vpe), self.edges_for(dorig['verts_rhand'], self.vpe))
        ########## KL loss
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([self.cfg.batch_size, self.cfg.latentD]), requires_grad=False).to(device).type(dtype),
            scale=torch.tensor(np.ones([self.cfg.batch_size, self.cfg.latentD]), requires_grad=False).to(device).type(dtype))
        loss_kl = self.cfg.kl_coef * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))
        ##########

        loss_dict = {'loss_kl': loss_kl,
                     'loss_edge': loss_edge,
                     'loss_mesh_rec': loss_mesh_rec_w,
                     'loss_dist_h': loss_dist_h,
                     'loss_dist_o': loss_dist_o,
                     }
        
        ########## direction loss
        #if self.cfg.fix_dir and not self.cfg.dir_adapted:
        dir_rhand = verts_rhand[:, 220] - verts_rhand[:, 216]
        dir_rhand = dir_rhand / (((dir_rhand**2).sum(1)).sqrt()[..., None] + 1e-8)

        #loss_dir = 1 * (1. - self.cfg.kl_coef) * self.LossL2(dorig['dir_norm'], dir_rhand) # L2 distance
        loss_dir = (1 - self.cfg.kl_coef) * self.LossL2(dorig['dir_hand'], dir_rhand) # L2 distance
        loss_dict['loss_dir'] = loss_dir

        ####### feature loss
        rh2obj_gt = dorig['verts2obj']
        loss_features = (1 - self.cfg.kl_coef) * self.LossL1(rh2obj_gt, drec['dist'].reshape(rh2obj_gt.shape))
        loss_dict['loss_features'] = loss_features
        ##########

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    def fit(self, n_epochs=None, message=None):

        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))
        if message is not None:
            self.logger(message)

        prev_lr_cgrasp = np.inf

        lr_scheduler_cgrasp = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_cgrasp, 'min')
        early_stopping_cgrasp = EarlyStopping(patience=8, trace_func=self.logger)

        for epoch_num in range(1, n_epochs + 1):
            self.logger('--- starting Epoch # %03d' % epoch_num)

            if self.cfg.log:
                wandb.log({"epoch": epoch_num})

            train_loss_dict_cgrasp = self.train()
            eval_loss_dict_cgrasp = self.evaluate()



            if self.cfg.log:
                wandb.log({"train loss cgrasp": train_loss_dict_cgrasp['loss_total']})
                wandb.log({"eval loss cgrasp": eval_loss_dict_cgrasp['loss_total']})

            lr_scheduler_cgrasp.step(eval_loss_dict_cgrasp['loss_total'])
            cur_lr_cgrasp = self.optimizer_cgrasp.param_groups[0]['lr']

            if cur_lr_cgrasp != prev_lr_cgrasp:
                self.logger('--- CGrasp learning rate changed from %.2e to %.2e ---' % (prev_lr_cgrasp, cur_lr_cgrasp))
                prev_lr_cgrasp = cur_lr_cgrasp

            with torch.no_grad():
                eval_msg = Trainer.create_loss_message(eval_loss_dict_cgrasp, expr_ID=self.cfg.expr_ID,
                                                        epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                        model_name='CGrasp',
                                                        try_num=self.try_num, mode='evald')
                if eval_loss_dict_cgrasp['loss_total'] < self.best_loss_cgrasp:

                    self.cfg.cgrasp_model_path = makepath(os.path.join(self.cfg.save_dir, 'snapshots', 'TR%02d_E%03d_cgrasp.pt' % (self.try_num, self.epochs_completed)), isfile=True)
                    self.save_cgrasp()
                    self.logger(eval_msg + ' ** ')
                    self.best_loss_cgrasp = eval_loss_dict_cgrasp['loss_total']

                else:
                    self.logger(eval_msg)

                self.swriter.add_scalars('total_loss_cgrasp/scalars',
                                         {'train_loss_total': train_loss_dict_cgrasp['loss_total'],
                                         'evald_loss_total': eval_loss_dict_cgrasp['loss_total'], },
                                         self.epochs_completed)

                if early_stopping_cgrasp(eval_loss_dict_cgrasp['loss_total']):
                    self.logger('Early stopping CGrasp training!')
                    break

            self.epochs_completed += 1

        self.logger('Stopping the training!')

        endtime = datetime.now().replace(microsecond=0)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger(
            'Training done in %s!\n' % (endtime - starttime))
        self.logger('Best CGrasp val total loss achieved: %.2e\n' % (self.best_loss_cgrasp))
        self.logger('Best CGrasp model path: %s\n' % self.cfg.cgrasp_model_path)


    @staticmethod
    def create_loss_message(loss_dict, expr_ID='XX', epoch_num=0,model_name='CGrasp', it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s - %s: [T:%.2e] - [%s]' % (
            expr_ID, try_num, epoch_num, it,model_name, mode, loss_dict['loss_total'], ext_msg)
