from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from ikol.utils.render import SMPLRenderer
from ikol.utils.vis import get_one_box, vis_bbox, vis_smpl_3d

from .lbs import lbs, lbs_generation, lbs_Twist_leaf, hybrik, hybrik_naive, rotmat_to_quat, quat_to_rotmat, blend_shapes, vertices2joints, batch_get_pelvis_orient_svd, batch_rigid_transform, batch_get_3children_orient_svd,batch_get_pelvis_orient, transform_mat
import time
import math

try:
    import cPickle as pk
except ImportError:
    import pickle as pk
from torch.autograd.functional import jacobian, hessian
# from functorch import jacrev, vmap

ModelOutput = namedtuple('ModelOutput',
                         ['vertices', 'joints', 'joints_from_verts',
                          'rot_mats'])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

def batch_jacobian(f, x):
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return jacobian(f_sum, x)

class SMPL_layer(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = [
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb',            # 23
        'head', 'left_middle', 'right_middle',  # 26
        'left_bigtoe', 'right_bigtoe'           # 28
    ]
    LEAF_NAMES = [
        'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe'
    ]

    # leaf 15, 22, 23, 10, 11
    root_idx_17 = 0
    root_idx_smpl = 0

    def __init__(self,
                 model_path,
                 h36m_jregressor,
                 gender='neutral',
                 dtype=torch.float32,
                 num_joints=29):
        ''' SMPL model layers

        Parameters:
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        gender: str, optional
            Which gender to load
        '''
        super(SMPL_layer, self).__init__()

        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        self.LEAF_IDX = [self.JOINT_NAMES.index(name) for name in self.LEAF_NAMES]
        self.SPINE3_IDX = 9

        with open(model_path, 'rb') as smpl_file:
            self.smpl_data = Struct(**pk.load(smpl_file, encoding='latin1'))

        self.gender = gender

        self.dtype = dtype

        self.faces = self.smpl_data.f

        ''' Register Buffer '''
        # Faces
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.smpl_data.f, dtype=np.int64), dtype=torch.long))

        # The vertices of the template model, (6890, 3)
        self.register_buffer('v_template',
                             to_tensor(to_np(self.smpl_data.v_template), dtype=dtype))

        # The shape components
        # Shape blend shapes basis, (6890, 3, 10)
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(self.smpl_data.shapedirs), dtype=dtype))

        # Pose blend shape basis: 6890 x 3 x 23*9, reshaped to 6890*3 x 23*9
        num_pose_basis = self.smpl_data.posedirs.shape[-1]
        # 23*9 x 6890*3
        posedirs = np.reshape(self.smpl_data.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # Vertices to Joints location (23 + 1, 6890)
        self.register_buffer(
            'J_regressor',
            to_tensor(to_np(self.smpl_data.J_regressor), dtype=dtype))
        # Vertices to Human3.6M Joints location (17, 6890)
        self.register_buffer(
            'J_regressor_h36m',
            to_tensor(to_np(h36m_jregressor), dtype=dtype))

        self.num_joints = num_joints

        # indices of parents for each joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        parents[:(self.NUM_JOINTS + 1)] = to_tensor(to_np(self.smpl_data.kintree_table[0])).long()
        parents[0] = -1
        # extend kinematic tree
        parents[24] = 15
        parents[25] = 22
        parents[26] = 23
        parents[27] = 10
        parents[28] = 11
        if parents.shape[0] > self.num_joints:
            parents = parents[:24]

        self.register_buffer(
            'children_map',
            self._parents_to_children(parents))

        self.register_buffer(
            'children_map_opt',
            self._parents_to_children_opt(parents))

        # (24,)
        self.register_buffer('parents', parents)

        # (6890, 23 + 1)
        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.smpl_data.weights), dtype=dtype))

        self.idx_levs = [
            [0],  # 0
            [3],  # 1
            [6],  # 2
            [9],  # 3
            [1, 2, 12, 13, 14],  # 4
            [4, 5, 16, 17],  # 5
            [7, 8, 18, 19],  # 6
            [20, 21],  # 7
            [15, 22, 23, 10, 11]  # 8
            ]

        self.parent_indexs = [
            [-1],  # 0
            [-1],  # 1
            [-1],  # 2
            [-1],  # 3
            [0, 1],  # 4
            [0, 2],  # 5
            [0, 3],  # 6
            [0, 1, 4],  # 7
            [0, 2, 5],  # 8
            [0, 3, 6],  # 9
            [0, 1, 4, 7],  # 10
            [0, 2, 5, 8],  # 11
            [0, 3, 6, 9],  # 12
            [0, 3, 6, 9],  # 13
            [0, 3, 6, 9],  # 14
            [0, 3, 6, 9, 12],  # 15
            [0, 3, 6, 9, 13],  # 16
            [0, 3, 6, 9, 14],  # 17
            [0, 3, 6, 9, 13, 16],  # 18
            [0, 3, 6, 9, 14, 17],  # 19
            [0, 3, 6, 9, 13, 16, 18],  # 20
            [0, 3, 6, 9, 14, 17, 19],  # 21
            [0, 3, 6, 9, 13, 16, 18, 20],  # 22
            [0, 3, 6, 9, 14, 17, 19, 21]  # 23
            ]  # Affected parent node index

        self.idx_jacobian = [
            [4, 5, 6],
            [7, 8, 9],
            [10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19],
            [20, 21],
            [22, 23]
        ]  #  exclude 0,1,2,3

        self.index_18_to_24 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children


    def _parents_to_children_opt(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        # children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[-1] = -1
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children


    def forward(self,
                pose_axis_angle,
                betas,
                global_orient,
                transl=None,
                return_verts=True):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            pose_axis_angle: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        '''
        # batch_size = pose_axis_angle.shape[0]

        # concate root orientation with thetas
        if global_orient is not None:
            full_pose = torch.cat([global_orient, pose_axis_angle], dim=1)
        else:
            full_pose = pose_axis_angle

        # Translate thetas to rotation matrics
        pose2rot = True
        # vertices: (B, N, 3), joints: (B, K, 3)
        vertices, joints, rot_mats, joints_from_verts_h36m = lbs(betas, full_pose, self.v_template,
                                                                 self.shapedirs, self.posedirs,
                                                                 self.J_regressor, self.J_regressor_h36m, self.parents,
                                                                 self.lbs_weights, pose2rot=pose2rot, dtype=self.dtype)

        if transl is not None:
            # apply translations
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts_h36m += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(1).detach()
            joints = joints - joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts_h36m = joints_from_verts_h36m - joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = ModelOutput(
            vertices=vertices, joints=joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts_h36m)
        return output


    def forward_global_orient(self,
                pose_skeleton,
                rest_J):

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()  # 防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        if self.training:
            global_orient_mat = batch_get_pelvis_orient(
                rel_pose_skeleton.clone(), rel_rest_pose.clone(), self.parents, self.children_map, self.dtype)
        else:
            global_orient_mat = batch_get_pelvis_orient_svd(
                rel_pose_skeleton.clone(), rel_rest_pose.clone(), self.parents, self.children_map, self.dtype)

        return global_orient_mat


    def forward_rest_J(self, betas):
        v_shaped =  self.v_template + blend_shapes(betas, self.shapedirs)
        rest_J = vertices2joints(self.J_regressor, v_shaped)
        return rest_J, v_shaped


    def forward_jacobian_and_pred_test_efficient(self,
                                  pose_axis_angle,
                                  pose_skeleton,
                                  rest_J,
                                  global_orient,
                                  rot_mat_twist=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        global_orient_mat = global_orient
        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        jacobian = torch.zeros([batch_size, 18, 24, 3], dtype=torch.float32, device=device)
        R_derivative = torch.zeros((batch_size, 24, 24, 3, 3), dtype=torch.float32,
                                   device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1
        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
        index_24_to_18 = torch.tensor(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])

        for idx_lev in range(len(self.idx_jacobian)):
            indices = self.idx_jacobian[idx_lev]
            len_indices = len(indices)
            parents_1 = self.parents[indices]
            parents_2 = self.parents[parents_1]
            parents_3 = self.parents[parents_2]
            parents_4 = self.parents[parents_3]
            parents_5 = self.parents[parents_4]
            parents_6 = self.parents[parents_5]
            parents_7 = self.parents[parents_6]


            orig_vec_unrotate = rel_pose_skeleton[:, indices]
            if idx_lev == 3:
                spine_child = indices
                children_final_loc = []
                children_rest_loc = []
                for c in spine_child:
                    temp = rel_pose_skeleton[:, c]
                    children_final_loc.append(temp)
                    children_rest_loc.append(rel_rest_pose[:, c].clone())

                rot_mat = batch_get_3children_orient_svd(
                    children_final_loc, children_rest_loc,
                    rot_mat_chain[:, parents_2[0]], spine_child, self.dtype)

                rot_mat_chain[:, parents_1[0]] = torch.matmul(
                    rot_mat_chain[:, parents_2[0]],
                    rot_mat
                )

                rot_mat_local[:, parents_1[0]] = rot_mat
                orig_vec_unrotate = 0 * orig_vec_unrotate

                rotate_rest_pose[:, parents_1] = rotate_rest_pose[:, parents_2] + torch.matmul(
                    rot_mat_chain[:, parents_2],
                    rel_rest_pose[:, parents_1]
                )

                orig_vec = torch.matmul(
                    rot_mat_chain[:, parents_2].transpose(2, 3),
                    orig_vec_unrotate
                )
                child_rest_loc = rel_rest_pose[:, indices]  # need rotation back ?
                w = torch.cross(child_rest_loc, orig_vec, dim=2)
                w_norm = torch.norm(w, dim=2, keepdim=True)
                cos = pose_axis_angle[:, index_24_to_18[parents_1]].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
                sin = pose_axis_angle[:, index_24_to_18[parents_1]].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
                axis = w / (w_norm + 1e-8)
                rx, ry, rz = torch.split(axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
                rot_mat_spin = rot_mat_twist[:, parents_1]

            else:
                rotate_rest_pose[:, parents_1] = rotate_rest_pose[:, parents_2] + torch.matmul(
                    rot_mat_chain[:, parents_2],
                    rel_rest_pose[:, parents_1]
                )

                orig_vec = torch.matmul(
                    rot_mat_chain[:, parents_2].transpose(2, 3),
                    orig_vec_unrotate
                )
                child_rest_loc = rel_rest_pose[:, indices]
                w = torch.cross(child_rest_loc, orig_vec, dim=2)
                w_norm = torch.norm(w, dim=2, keepdim=True)
                cos = pose_axis_angle[:, index_24_to_18[parents_1]].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
                sin = pose_axis_angle[:, index_24_to_18[parents_1]].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)

                axis = w / (w_norm + 1e-8)
                rx, ry, rz = torch.split(axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
                rot_mat_spin = rot_mat_twist[:, parents_1]
                rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
                rot_mat_chain[:, parents_1] = torch.matmul(
                    rot_mat_chain[:, parents_2],
                    rot_mat)
                rot_mat_local[:, parents_1] = rot_mat

            q_idex = indices
            q_idex_child = self.parents[q_idex]
            tree_len = len(self.parent_indexs[indices[0]])
            # parent_index =  [self.parent_indexs[indices[i]] for i in range(tree_len)]
            parent_index = torch.tensor([self.parent_indexs[indices[i]] for i in range(len_indices)])

            for x_count in range(tree_len - 1):

                if x_count == tree_len - 2:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......
                    DR_sw = cos * K + sin * torch.matmul(K, K)
                    R_derivative[:, parents_1, parents_1] = torch.matmul(DR_sw,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    DR_ = torch.matmul(rot_mat_chain[:, parents_2], R_derivative[:, parents_1, parents_1])
                    Dq_1 = torch.matmul(DR_, child_rest_loc)
                    jacobian[:, index_24_to_18[parents_1], q_idex] = Dq_1[:, :, :, 0]

                elif x_count == tree_len - 3:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                    DR_k_1_k_1 = R_derivative[:, parents_2, parents_2]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_1
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn_k = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / ((w_norm + 1e-8) ** 3), Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2).view((batch_size, len_indices, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_2] = torch.matmul(DR_sw_k,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_2]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    Dq_k_1_k_1 = jacobian[:, index_24_to_18[parents_2], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1
                    Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                    jacobian[:, index_24_to_18[parents_2], q_idex] = Dq_k_k_1[:, :, :, 0]

                elif x_count == tree_len - 4:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......

                    DR_k_2_k_2 = R_derivative[:, parents_3, parents_3]
                    DR_k_1_k_2 = R_derivative[:, parents_2, parents_3]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()

                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_2
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_2

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_3]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])

                    Dq_k_1_k_2 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2
                    Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2
                    jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_2[:, :, :, 0]

                elif x_count == tree_len - 5:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                    DR_k_3_k_3 = R_derivative[:, parents_4, parents_4]
                    DR_k_2_k_3 = R_derivative[:, parents_3, parents_4]
                    DR_k_1_k_3 = R_derivative[:, parents_2, parents_4]

                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()

                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_3
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_3
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_3

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_4]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    Dq_k_1_k_3 = jacobian[:, index_24_to_18[parents_4], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3
                    Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                    jacobian[:, index_24_to_18[parents_4], q_idex] = Dq_k_k_3[:, :, :, 0]

                elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_4_k_4 = R_derivative[:, parents_5, parents_5]
                    DR_k_3_k_4 = R_derivative[:, parents_4, parents_5]
                    DR_k_2_k_4 = R_derivative[:, parents_3, parents_5]
                    DR_k_1_k_4 = R_derivative[:, parents_2, parents_5]

                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()

                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_4
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_4
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_4
                    rot_mat_local_withDR4[:, :, -5] = DR_k_4_k_4

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, :, 0]
                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_5]
                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, :, index_r])
                    Dq_k_1_k_4 = jacobian[:, index_24_to_18[parents_5], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4
                    jacobian[:, index_24_to_18[parents_5], q_idex] = Dq_k_k_4[:, :, :, 0]

                elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_5_k_5 = R_derivative[:, parents_6, parents_6]
                    DR_k_4_k_5 = R_derivative[:, parents_5, parents_6]
                    DR_k_3_k_5 = R_derivative[:, parents_4, parents_6]
                    DR_k_2_k_5 = R_derivative[:, parents_3, parents_6]
                    DR_k_1_k_5 = R_derivative[:, parents_2, parents_6]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR5 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_5
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_5
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_5
                    rot_mat_local_withDR4[:, :, -5] = DR_k_4_k_5
                    rot_mat_local_withDR5[:, :, -6] = DR_k_5_k_5
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, :, 0]
                    rot_mat_withDR5 = rot_mat_local_withDR5[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, :, index_r])
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                       rot_mat_local_withDR5[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3) + rot_mat_withDR5.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_6]
                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, :, index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, :, index_r])
                    Dq_k_1_k_5 = jacobian[:, index_24_to_18[parents_6], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5
                    jacobian[:, index_24_to_18[parents_6], q_idex] = Dq_k_k_4[:, :, :, 0]

                elif x_count == tree_len - 8:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_6_k_6 = R_derivative[:, parents_7, parents_7]
                    DR_k_5_k_6 = R_derivative[:, parents_6, parents_7]
                    DR_k_4_k_6 = R_derivative[:, parents_5, parents_7]
                    DR_k_3_k_6 = R_derivative[:, parents_4, parents_7]
                    DR_k_2_k_6 = R_derivative[:, parents_3, parents_7]
                    DR_k_1_k_6 = R_derivative[:, parents_2, parents_7]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR5 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR6 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_6
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_6
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_6
                    rot_mat_local_withDR4[:, :, -5] = DR_k_4_k_6
                    rot_mat_local_withDR5[:, :, -6] = DR_k_5_k_6
                    rot_mat_local_withDR6[:, :, -7] = DR_k_6_k_6
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, :, 0]
                    rot_mat_withDR5 = rot_mat_local_withDR5[:, :, 0]
                    rot_mat_withDR6 = rot_mat_local_withDR6[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, :, index_r])
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                       rot_mat_local_withDR5[:, :, index_r])
                        rot_mat_withDR6 = torch.matmul(rot_mat_withDR6,
                                                       rot_mat_local_withDR6[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3) + rot_mat_withDR5.transpose(2,
                                                                                            3) + rot_mat_withDR6.transpose(
                        2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_7] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_7]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, :, index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, :, index_r])
                    rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local_withDR6[:, :, index_r])
                    Dq_k_1_k_6 = jacobian[:, index_24_to_18[parents_7], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6
                    Dq_k_k_5 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6
                    jacobian[:, index_24_to_18[parents_7], q_idex] = Dq_k_k_5[:, :, :, 0]

        leaf_index = [10, 11, 15, 22, 23]
        rotate_rest_pose[:, leaf_index] = rotate_rest_pose[:, self.parents[leaf_index]] + torch.matmul(
            rot_mat_chain[:, self.parents[leaf_index]],
            rel_rest_pose[:, leaf_index]
        )
        rotate_rest_pose = rotate_rest_pose.squeeze(-1).contiguous()
        new_joints = rotate_rest_pose - rotate_rest_pose[:, self.root_idx_smpl, :].unsqueeze(1).detach()


        return jacobian.reshape(batch_size, 18, 24 * 3).transpose(1, 2), new_joints, rot_mat_local


    def forward_jacobian_autograd_test(self,
                                           pose_axis_angle,
                                           pose_skeleton,
                                           rest_J,
                                           global_orient,
                                           rot_mat_twist=None,
                                           rotmat_leaf=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] = rel_rest_pose[:, 1:] - rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        # the predicted final pose
        final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)

        rot_mat_chain = [global_orient]
        rot_mat_local = [global_orient]

        index_24_to_18 = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])
        leaf_cnt = 0
        for indices in range(1, 24):
            parents_1 = self.parents[indices]
            children_1 = self.children_map_opt[indices]

            if children_1 == -1:
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                    rot_mat_chain[parents_1],
                    rel_rest_pose[:, indices]
                )
                rot_mat = rotmat_leaf[:, leaf_cnt, :, :]
                leaf_cnt += 1
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents_1],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            elif indices == 9:
                # (B, 3, 1)
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                    rot_mat_chain[parents_1],
                    rel_rest_pose[:, indices]
                )

                spine_child = []
                for c in range(1, self.parents.shape[0]):
                    if self.parents[c] == indices and c not in spine_child:
                        spine_child.append(c)

                children_final_loc = []
                children_rest_loc = []
                for c in spine_child:
                    temp = rel_pose_skeleton[:, c].clone()
                    children_final_loc.append(temp)
                    children_rest_loc.append(rel_rest_pose[:, c].clone())

                rot_mat = batch_get_3children_orient_svd(
                    children_final_loc, children_rest_loc,
                    rot_mat_chain[parents_1], spine_child, self.dtype)

                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents_1],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            else:
                # (B, 3, 1)
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                    rot_mat_chain[parents_1],
                    rel_rest_pose[:, indices]
                )

                orig_vec_unrotate = rel_pose_skeleton[:, children_1]

                orig_vec = torch.matmul(
                    rot_mat_chain[parents_1].transpose(1, 2),
                    orig_vec_unrotate
                )

                child_rest_loc = rel_rest_pose[:, children_1]  # need rotation back ?
                child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)
                # (B, K, 3, 1)
                w = torch.cross(child_rest_loc, orig_vec, dim=1)
                w_norm = torch.norm(w, dim=1, keepdim=True)
                # (B, K, 1, 1)
                cos = pose_axis_angle[:, index_24_to_18[indices]].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
                sin = pose_axis_angle[:, index_24_to_18[indices]].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)

                axis = w / (w_norm + 1e-8)
                # Convert location revolve to rot_mat by rodrigues
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(axis, 1, dim=1)
                zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                    .view((batch_size, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)
                rot_mat_spin = rot_mat_twist[:, indices]
                rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents_1],
                    rot_mat))
                rot_mat_local.append(rot_mat)


        rot_mats = torch.stack(rot_mat_local, dim=1)

        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents, dtype=self.dtype)
        new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()

        return new_joints.reshape(-1, 72), rot_mats

    def forward_jacobian_autograd_train(self,
                                           pose_axis_angle,
                                           pose_skeleton,
                                           rest_J,
                                           global_orient,
                                           rot_mat_twist=None,
                                           rotmat_leaf=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] = rel_rest_pose[:, 1:] - rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        # the predicted final pose
        final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
        final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, 0:1] + rel_rest_pose[:, 0:1]

        rot_mat_chain = [global_orient]
        rot_mat_local = [global_orient]
        index_24_to_18 = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])
        leaf_cnt = 0
        for indices in range(1, 24):
            # indices = idx_lev
            parents_1 = self.parents[indices]  ## 父节点
            children_1 = self.children_map_opt[indices]

            if children_1 == -1:
                # (B, 3, 1)
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                    rot_mat_chain[parents_1],
                    rel_rest_pose[:, indices]
                )
                rot_mat = rotmat_leaf[:, leaf_cnt, :, :]
                leaf_cnt += 1
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents_1],
                    rot_mat))
                rot_mat_local.append(rot_mat)
            else:
                # (B, 3, 1)
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                    rot_mat_chain[parents_1],
                    rel_rest_pose[:, indices]
                )

                orig_vec_unrotate = rel_pose_skeleton[:, children_1]

                orig_vec = torch.matmul(
                    rot_mat_chain[parents_1].transpose(1, 2),
                    orig_vec_unrotate
                )

                child_rest_loc = rel_rest_pose[:, children_1]
                w = torch.cross(child_rest_loc, orig_vec, dim=1)
                w_norm = torch.norm(w, dim=1, keepdim=True)
                cos = pose_axis_angle[:, index_24_to_18[indices]].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
                sin = pose_axis_angle[:, index_24_to_18[indices]].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
                axis = w / (w_norm + 1e-8)
                # Convert location revolve to rot_mat by rodrigues
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(axis, 1, dim=1)
                zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                    .view((batch_size, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)
                rot_mat_spin = rot_mat_twist[indices]
                rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents_1],
                    rot_mat))
                rot_mat_local.append(rot_mat)

        rot_mats = torch.stack(rot_mat_local, dim=1)

        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents, dtype=self.dtype)
        new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()

        return new_joints.reshape(-1, 72), rot_mats

    def forward_full_withtwist_test(self,
                                   pose_axis_angle,
                                   pose_skeleton,
                                   rest_J,
                                   global_orient,
                                   rot_mat_twist=None,
                                   rotmat_leaf=None,
                                   v_shaped=None,
                                   transl=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device


        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)
        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        global_orient_mat = global_orient

        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
        index_24_to_18 = torch.tensor(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])

        for idx_lev in range(len(self.idx_jacobian)):
            indices = self.idx_jacobian[idx_lev]
            len_indices = len(indices)
            parents_1 = self.parents[indices]
            parents_2 = self.parents[parents_1]
            if idx_lev == 3:
                # original
                spine_child = indices
                children_final_loc = []
                children_rest_loc = []
                for c in spine_child:
                    temp = rel_pose_skeleton[:, c]
                    children_final_loc.append(temp)
                    children_rest_loc.append(rel_rest_pose[:, c].clone())

                rot_mat = batch_get_3children_orient_svd(
                    children_final_loc, children_rest_loc,
                    rot_mat_chain[:, parents_2[0]], spine_child, self.dtype)
                rot_mat_chain[:, parents_1[0]] = torch.matmul(
                    rot_mat_chain[:, parents_2[0]],
                    rot_mat
                )
                rot_mat_local[:, parents_1[0]] = rot_mat
            else:
                orig_vec_unrotate = rel_pose_skeleton[:, indices]
                orig_vec = torch.matmul(
                    rot_mat_chain[:, parents_2].transpose(2, 3),
                    orig_vec_unrotate
                )
                child_rest_loc = rel_rest_pose[:, indices]  # need rotation back ?
                # (B, K, 3, 1)
                w = torch.cross(child_rest_loc, orig_vec, dim=2)
                w_norm = torch.norm(w, dim=2, keepdim=True)
                # (B, K, 1, 1)
                cos = pose_axis_angle[:, index_24_to_18[parents_1]].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
                sin = pose_axis_angle[:, index_24_to_18[parents_1]].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
                # (B, K, 3, 1)
                axis = w / (w_norm + 1e-8)
                # Convert location revolve to rot_mat by rodrigues
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
                rot_mat_spin = rot_mat_twist[:, parents_1]
                rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
                rot_mat_chain[:, parents_1] = torch.matmul(
                    rot_mat_chain[:, parents_2],
                    rot_mat)
                rot_mat_local[:, parents_1] = rot_mat


        rot_mats = rot_mat_local
        parent_indices = [10, 11, 15, 22, 23]
        rot_mats[:, parent_indices] = rotmat_leaf
        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents, dtype=self.dtype)

        ident = torch.eye(3, dtype=self.dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, self.posedirs).view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped

        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints_from_verts = vertices2joints(self.J_regressor_h36m, vertices)

        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        rot_mats = rotmat_to_quat(rot_mats).reshape(batch_size, 24 * 4)

        if transl is not None:
            J_transformed += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = ModelOutput(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)


        return output


    def forward_full_withtwist_train(self,
                                   pose_axis_angle,
                                   pose_skeleton,
                                   rest_J,
                                   global_orient,
                                   rot_mat_twist=None,
                                   rotmat_leaf=None,
                                   v_shaped=None,
                                   transl=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] = rel_rest_pose[:, 1:] - rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        # the predicted final pose
        final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)

        rot_mat_chain = [global_orient]
        rot_mat_local = [global_orient]
        index_24_to_18 = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])
        leaf_cnt = 0
        for indices in range(1, 24):
            parents_1 = self.parents[indices]  ## 父节点
            children_1 = self.children_map_opt[indices]

            if children_1 == -1:
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                    rot_mat_chain[parents_1],
                    rel_rest_pose[:, indices]
                )
                rot_mat = rotmat_leaf[:, leaf_cnt, :, :]
                leaf_cnt += 1
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents_1],
                    rot_mat))
                rot_mat_local.append(rot_mat)
            else:

                rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                    rot_mat_chain[parents_1],
                    rel_rest_pose[:, indices]
                )

                orig_vec_unrotate = rel_pose_skeleton[:, children_1]

                orig_vec = torch.matmul(
                    rot_mat_chain[parents_1].transpose(1, 2),
                    orig_vec_unrotate
                )

                child_rest_loc = rel_rest_pose[:, children_1]  # need rotation back ?
                # child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)
                # (B, K, 3, 1)
                w = torch.cross(child_rest_loc, orig_vec, dim=1)
                w_norm = torch.norm(w, dim=1, keepdim=True)
                # (B, K, 1, 1)
                cos = pose_axis_angle[:, index_24_to_18[indices]].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
                sin = pose_axis_angle[:, index_24_to_18[indices]].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)

                # (B, K, 3, 1)
                axis = w / (w_norm + 1e-8)
                # Convert location revolve to rot_mat by rodrigues
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(axis, 1, dim=1)
                zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                    .view((batch_size, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

                rot_mat_spin = rot_mat_twist[indices]
                rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents_1],
                    rot_mat))
                rot_mat_local.append(rot_mat)

        rot_mats = torch.stack(rot_mat_local, dim=1)
        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents, dtype=self.dtype)

        ident = torch.eye(3, dtype=self.dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, self.posedirs).view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped

        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints_from_verts = vertices2joints(self.J_regressor_h36m, vertices)

        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        rot_mats = rotmat_to_quat(rot_mats).reshape(batch_size, 24 * 4)

        if transl is not None:
            J_transformed += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = ModelOutput(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)

        return output

    def forward_twist_and_leaf_test(self,
                rest_J,
                phis,
                global_orient,
                leaf_thetas=None):

        batch_size = rest_J.shape[0]
        device = rest_J.device

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)
        phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)
        global_orient_mat = global_orient
        rot_mat_local = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local[:, 0] = global_orient_mat


        leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
        rotmat_leaf = quat_to_rotmat(leaf_thetas)
        rotmat_leaf_ = rotmat_leaf.view([batch_size, 5, 3, 3])

        for idx_lev in range(1, len(self.idx_levs)-1):
            indices = self.idx_levs[idx_lev]
            len_indices = len(indices)
            child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]
            child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)
            spin_axis = child_rest_loc / child_rest_norm
            rx, ry, rz = torch.split(spin_axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
            phi_indices = [item - 1 for item in indices]
            cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
            cos = torch.unsqueeze(cos, dim=3)
            sin = torch.unsqueeze(sin, dim=3)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            rot_mat_local[:, indices] = rot_mat_spin

        return rot_mat_local, rotmat_leaf_

    def forward_twist_and_leaf_train(self,
                rest_J,
                phis,
                global_orient,
                leaf_thetas=None):

        batch_size = rest_J.shape[0]
        device = rest_J.device

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)
        phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)
        rot_mat_local = [global_orient]

        leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
        rotmat_leaf = quat_to_rotmat(leaf_thetas)
        rotmat_leaf_ = rotmat_leaf.view([batch_size, 5, 3, 3])

        for indices in range(1, 24):

            child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]
            # (B, K, 1, 1)
            child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)
            # Convert spin to rot_mat
            # (B, K, 3, 1)
            spin_axis = child_rest_loc / (child_rest_norm + 1e-8)
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
            # (B, K, 1, 1)
            cos, sin = torch.split(phis[:, indices - 1], 1, dim=1)
            cos = torch.unsqueeze(cos, dim=2)
            sin = torch.unsqueeze(sin, dim=2)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.bmm(K, K)
            rot_mat_local.append(rot_mat_spin)

        return rot_mat_local, rotmat_leaf_


    def single_iteration_test_efficient(self,
                         pose_axis_angle,
                         target,
                         rest_J,
                         global_orient,
                         rot_mat_twist=None,
                         u=None):

        batch_size = target.shape[0]

        jacobian, pred, rot_mat = self.forward_jacobian_and_pred_test_efficient(
            pose_axis_angle=pose_axis_angle,
            pose_skeleton=target,
            rest_J=rest_J,
            global_orient=global_orient,
            rot_mat_twist=rot_mat_twist
        )

        residual = (pred - target).view(batch_size, -1).unsqueeze(-1)
        mse = residual.square().mean(1).squeeze()
        jtj = torch.bmm(jacobian.transpose(2, 1), jacobian, out=None)
        ident = torch.eye(18).cuda().reshape(1, 18, 18).repeat(batch_size, 1, 1)
        jtj = jtj + u * ident

        delta = torch.bmm(
            torch.bmm(jtj.inverse(), jacobian.transpose(2, 1)), residual
        ).squeeze()

        return (pose_axis_angle-delta), mse, rot_mat

    def single_iteration_test(self,
                         pose_axis_angle,
                         target,
                         rest_J,
                         global_orient,
                         rot_mat_twist=None,
                         rotmat_leaf=None,
                         u=None):


        batch_size = target.shape[0]
        device = target.device


        new_joints, rot_mats = self.forward_jacobian_autograd_test(
            pose_axis_angle=pose_axis_angle,
            pose_skeleton=target.clone(),
            rest_J=rest_J.clone(),
            global_orient=global_orient,
            rot_mat_twist=rot_mat_twist,
            rotmat_leaf=rotmat_leaf
        )

        f_sum = lambda pose_axis_angle: torch.sum(self.forward_jacobian_autograd_test(pose_axis_angle,target,rest_J,global_orient, rot_mat_twist, rotmat_leaf)[0], axis=0)
        jacobian_pose_axis_angle = jacobian(f_sum, (pose_axis_angle), create_graph=False, vectorize=True).transpose(0,1)

        residual = (new_joints - target.view(-1, 72)).unsqueeze(-1)
        mse = residual.square().mean(1).squeeze()
        jtj = torch.bmm(jacobian_pose_axis_angle.transpose(2, 1), jacobian_pose_axis_angle, out=None)
        ident = torch.eye(18, dtype=torch.float32, device=device).reshape(1, 18, 18).repeat(batch_size, 1, 1)
        jtj = jtj + u * ident
        delta = torch.bmm(
            torch.bmm(jtj.inverse(), jacobian_pose_axis_angle.transpose(2, 1)), residual
        ).squeeze()

        return (pose_axis_angle-delta), mse, rot_mats

    def single_iteration_train(self,
                         pose_axis_angle,
                         target,
                         rest_J,
                         global_orient,
                         rot_mat_twist=None,
                         rotmat_leaf=None,
                         u=None):


        batch_size = target.shape[0]
        device = target.device


        new_joints, rot_mats = self.forward_jacobian_autograd_train(
            pose_axis_angle=pose_axis_angle,
            pose_skeleton=target.clone(),
            rest_J=rest_J.clone(),
            global_orient=global_orient,
            rot_mat_twist=rot_mat_twist,
            rotmat_leaf=rotmat_leaf
        )

        f_sum = lambda pose_axis_angle: torch.sum(self.forward_jacobian_autograd_train(pose_axis_angle,target,rest_J,global_orient, rot_mat_twist, rotmat_leaf)[0], axis=0)
        jacobian_pose_axis_angle = jacobian(f_sum, (pose_axis_angle), create_graph=False, vectorize=True).transpose(0,1)


        residual = (new_joints - target.view(-1, 72)).unsqueeze(-1)
        mse = residual.square().mean(1).squeeze()
        # print(mse)
        jtj = torch.bmm(jacobian_pose_axis_angle.transpose(2, 1), jacobian_pose_axis_angle, out=None)
        ident = torch.eye(18, dtype=torch.float32, device=device).reshape(1, 18, 18).repeat(batch_size, 1, 1)
        jtj = jtj + u * ident
        delta = torch.bmm(
            torch.bmm(jtj.inverse(), jacobian_pose_axis_angle.transpose(2, 1)), residual
        ).squeeze()

        return (pose_axis_angle-delta), mse, rot_mats


    def IKOL(self, target, betas=None, pred_phi=None, pred_leaf=None, efficient=True):

        batch_size = target.shape[0]
        device = target.device
        rest_J, v_shaped = self.forward_rest_J(betas)
        global_orient = self.forward_global_orient(pose_skeleton=target, rest_J=rest_J)

        if self.training:
            rot_mat_twist, rotmat_leaf = self.forward_twist_and_leaf_train(rest_J=rest_J, phis=pred_phi, global_orient=global_orient, leaf_thetas=pred_leaf)
            namespace = globals()
            namespace['params%d' % 0] = torch.zeros([batch_size, 18], dtype=torch.float32, device=device)
            namespace['mse%d' % 0] = torch.zeros([batch_size], dtype=torch.float32, device=device)
            namespace['update%d' % 0] = torch.zeros([batch_size], dtype=torch.float32, device=device)
            namespace['u%d' % 0] = 1e-1 * torch.ones([batch_size, 18, 18], dtype=torch.float32, device=device)
            for i in range(5):
                namespace['params%d' % (i + 1)], namespace['mse%d' % (i + 1)], namespace[
                    'rot_mats%d' % (i + 1)] = self.single_iteration_train(pose_axis_angle=namespace['params%d' % i],
                                                                             target=target.clone(),
                                                                             rest_J=rest_J.clone(),
                                                                             global_orient=global_orient.clone(),
                                                                             rot_mat_twist=rot_mat_twist,
                                                                             rotmat_leaf=rotmat_leaf.clone(),
                                                                             u=namespace['u%d' % i])
                namespace['update%d' % (i + 1)] = namespace['mse%d' % i] - namespace['mse%d' % (i + 1)]
                namespace['u%d' % (i + 1)] = namespace['u%d' % i].clone()

            output = self.forward_full_withtwist_train(
                pose_axis_angle=namespace['params%d' % (i + 1)],
                pose_skeleton=target,
                rest_J=rest_J.clone(),
                global_orient=global_orient,
                rot_mat_twist=rot_mat_twist,
                rotmat_leaf=rotmat_leaf,
                v_shaped=v_shaped)
        else:

            u = 1e-1 * torch.ones([batch_size, 18, 18], dtype=torch.float32, device=device)
            params = torch.zeros([batch_size, 18], dtype=torch.float32, device=device)
            rot_mat_twist, rotmat_leaf = self.forward_twist_and_leaf_test(rest_J=rest_J, phis=pred_phi, global_orient=global_orient, leaf_thetas=pred_leaf)
            for i in range(5):
                if efficient:
                    params, mse, rot_mat = self.single_iteration_test_efficient(pose_axis_angle=params, target=target,
                                                                                 rest_J=rest_J, global_orient=global_orient,
                                                                                 rot_mat_twist=rot_mat_twist, u=u)
                else:
                    params, mse, rot_mat = self.single_iteration_test(pose_axis_angle=params, target=target,
                                                                      rest_J=rest_J, global_orient=global_orient,
                                                                      rot_mat_twist=rot_mat_twist,
                                                                      rotmat_leaf=rotmat_leaf, u=u)

            output = self.forward_full_withtwist_test(pose_axis_angle=params,
                                                                pose_skeleton=target,
                                                                rest_J=rest_J.clone(),
                                                                global_orient=global_orient,
                                                                rot_mat_twist=rot_mat_twist,
                                                                rotmat_leaf=rotmat_leaf,
                                                                v_shaped=v_shaped)

        return output