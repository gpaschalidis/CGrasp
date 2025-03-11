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
import numpy as np
import open3d as o3d
import torch
import mano
import os
import random

from bps_torch.bps import bps_torch
from cgrasp.tools.utils import euler
from cgrasp.tools.meshviewer import Mesh
from cgrasp.tools.train_tools import point2point_signed
from cgrasp.tools.utils import aa2rotmat
from cgrasp.tools.utils import to_cpu

from .vis_utils import read_o3d_mesh, create_o3d_mesh, create_line_set


class Grasp:
    def __init__(self, cgrasp_tester, obj_path, save_path, closed_mano_faces_path, scale=1):
        self.cgrasp_tester = cgrasp_tester
        self.obj_path = obj_path
        self.save_path = save_path
        self.scale = scale
        self.device = cgrasp_tester.device
        self.rotmat = None
        self.directions = None
        self.cgrasp = None
        self.refine_net = None
        self.rh_model = None
        self.n_samples = None
        self.grasp_type = "right"
        self.closed_mano_faces = dict(np.load(closed_mano_faces_path))
        self.dorig = {}


    def set_input_params(self, n_samples, rotmat=None, direction=None, grasp_type="right"):
        self.cgrasp_tester.cgrasp.eval()
        self.cgrasp = self.cgrasp_tester.cgrasp.to(self.device)
        self.cgrasp_tester.refine_net.eval()

        rh_model = mano.load(model_path=self.cgrasp_tester.cfg.rhm_path,
                     model_type='mano',
                     num_pca_comps=45,
                     batch_size=n_samples,
                     flat_hand_mean=True).to(self.device)

        self.rh_model = rh_model        
        
        self.cgrasp_tester.refine_net.rhm_train = rh_model
        self.refine_net = self.cgrasp_tester.refine_net.to(self.device)
        self.n_samples = n_samples
        self.grasp_type = grasp_type
        bps = bps_torch(custom_basis=self.cgrasp_tester.bps)
        # We want to save the initial rand_rotmat
    
        rand_rotmat = False
        if rotmat is None:
            rand_rotmat = True
            rotmat = np.random.random([n_samples, 3]) * np.array([360, 360, 360])
            rotmat = euler(rotmat)
        self.rotmat = rotmat.copy()
        
   
        if direction is None:
            direction = np.array([self.generate_random_vector() for i in range(n_samples)])
        
        if grasp_type == "left":
            M = np.eye(3)
            M[0][0] = -1
            # We mirror the global orientation of the object
            rotmat = M @ rotmat @ M 
            direction = direction @ M.T
           
        if not rand_rotmat: 
            rotmat = [rotmat] * n_samples
        
        self.directions = torch.tensor(direction).to(torch.float32).to(self.device)
                    
        dorig = {'bps_object': [], 
                'verts_object': []} 

        for i in range(n_samples):
            verts_obj = self.load_obj_verts(
                obj_path=self.obj_path, rotmat=rotmat[i], scale=self.scale, grasp_type=grasp_type
            )
            bps_object = bps.encode(torch.from_numpy(verts_obj), feature_type='dists')['dists']

            dorig['bps_object'].append(bps_object.to(self.device))
            dorig['verts_object'].append(torch.from_numpy(verts_obj.astype(np.float32)).unsqueeze(0))

        dorig['bps_object'] = torch.cat(dorig['bps_object'])
        dorig['verts_object'] = torch.cat(dorig['verts_object'])
        self.dorig = dorig
         

    def load_obj_verts(self, obj_path, rotmat, scale, grasp_type="right", n_sample_verts=2048):
        np.random.seed(100)
        obj_mesh = Mesh(filename=obj_path, vscale=scale)

        ## center and scale the object
        max_length = np.linalg.norm(obj_mesh.vertices, axis=1).max()
        if max_length > .3:
            re_scale = max_length / .08
            print(f'The object is very large, down-scaling by {re_scale} factor')
            obj_mesh.vertices[:] = obj_mesh.vertices / re_scale

        object_fullpts = obj_mesh.vertices

        if grasp_type == "left": # We mirror the object vertices
            object_fullpts[:,0] *= -1

        maximum = object_fullpts.max(0, keepdims=True)
        minimum = object_fullpts.min(0, keepdims=True)

        offset = (maximum + minimum) / 2
        verts_obj = object_fullpts - offset
        obj_mesh.vertices[:] = verts_obj
        obj_mesh.rotate_vertices(rotmat)

        while (obj_mesh.vertices.shape[0]<n_sample_verts):
            new_mesh = obj_mesh.subdivide()
            obj_mesh = Mesh(vertices=new_mesh.vertices,
                            faces = new_mesh.faces,
                            visual = new_mesh.visual)

        verts_obj = obj_mesh.vertices
        verts_sample_id = np.random.choice(verts_obj.shape[0], n_sample_verts, replace=False)
        verts_sampled = verts_obj[verts_sample_id]

        return verts_sampled


    def generate_grasps(self):
        with torch.no_grad():
            cgrasp_out = self.cgrasp.sample_poses(self.dorig['bps_object'].to(self.device), dir_hand=self.directions)

            output = self.rh_model(**cgrasp_out)
            verts_rh_gen_cgrasp = output.vertices

            _, h2o, _ = point2point_signed(verts_rh_gen_cgrasp, self.dorig['verts_object'].to(self.device))
            cgrasp_out['trans_rhand_f'] = cgrasp_out['transl']
            cgrasp_out['global_orient_rhand_rotmat_f'] = aa2rotmat(cgrasp_out['global_orient']).view(-1, 3, 3)
            cgrasp_out['fpose_rhand_rotmat_f'] = aa2rotmat(cgrasp_out['hand_pose']).view(-1, 15, 3, 3)
            cgrasp_out['verts_object'] = self.dorig['verts_object'].to(self.device)
            cgrasp_out['h2o_dist'] = h2o.abs()

            drec_rnet = self.refine_net(**cgrasp_out)
            hand_verts = self.rh_model(**drec_rnet).vertices
            
            if self.grasp_type == "left":
                hand_verts[:,:,0] *= -1
 
            return drec_rnet, hand_verts

    def save_visualize_grasps(self, vis=True):
        _, hand_verts = self.generate_grasps()
        hand_verts = hand_verts.cpu().numpy()
        M = np.eye(3)
        if self.grasp_type == "right":
            hand_faces = self.closed_mano_faces["rh_faces"]
        elif self.grasp_type == "left":    
            hand_faces = self.closed_mano_faces["lh_faces"]
            M[0][0] = -1
        
        for i in range(0, self.n_samples):
            obj_mesh = read_o3d_mesh(self.obj_path)
            obj_verts = np.array(obj_mesh.vertices)
            obj_verts = obj_verts @ self.rotmat[i].T
            obj_faces = np.array(obj_mesh.triangles)
            obj_mesh = create_o3d_mesh(obj_verts, obj_faces, [0.9, 0.4, 0.3])
            hand_mesh = create_o3d_mesh(hand_verts[i], hand_faces, [0.8, 0.1, 0.3])
            origin  = (obj_verts.max(0) + obj_verts.min(0)) / 2
            target_dir = (self.directions[i].cpu().numpy() @ M)
            line = create_line_set(origin[None], origin[None] + 0.5 * target_dir, [0,0,1])
            if vis == True:
                o3d.visualization.draw_geometries([obj_mesh, hand_mesh, line])
       
        np.savez(os.path.join(self.save_path, "hand_verts.npz"),
                 hand_verts=hand_verts
        ) 
        


    def generate_random_vector(self):
        """
        Generates a random 3D vector.
        """
        vector = np.random.randn(3)
        while np.linalg.norm(vector) == 0:
            vector = np.random.randn(3)
        return vector / np.linalg.norm(vector)

































