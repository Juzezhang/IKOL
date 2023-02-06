import bisect
import random

from typing import Optional, Union
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler
from .builder import DATASETS, build_dataset

import cv2
from .h36m_smpl import H36mSMPL
from .hp3d import HP3D
from .mscoco import Mscoco
from .pw3d import PW3D
s_mpii_2_smpl_jt = [
    6, 3, 2,
    -1, 4, 1,
    -1, 5, 0,
    -1, -1, -1,
    8, -1, -1,
    -1,
    13, 12,
    14, 11,
    15, 10,
    -1, -1
]
s_3dhp_2_smpl_jt = [
    4, -1, -1,
    -1, 19, 24,
    -1, 20, 25,
    -1, -1, -1,  # TODO: foot point
    5, -1, -1,
    -1,
    9, 14,
    10, 15,
    11, 16,
    -1, -1
]
s_coco_2_smpl_jt = [
    -1, -1, -1,
    -1, 13, 14,
    -1, 15, 16,
    -1, -1, -1,
    -1, -1, -1,
    -1,
    5, 6,
    7, 8,
    9, 10,
    -1, -1
]

s_smpl24_jt_num = 24

h36m_idxs = [
    148, 145, 4, 7, 144, 5, 8, 150, 146, 152, 147, 16, 18, 20, 17, 19, 21
]
hybrik29_idxs = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22, 16, 17, 18, 19, 20,
    21, 66, 71, 15, 68, 73, 60, 63
]


data_keys = [
    'trans_inv', 'intrinsic_param', 'joint_root', 'depth_factor',
    'target_uvd_29', 'target_xyz_24', 'target_weight_24', 'target_weight_29',
    'target_xyz_17', 'target_weight_17', 'target_theta', 'target_beta',
    'target_smpl_weight', 'target_theta_weight', 'target_twist',
    'target_twist_weight', 'bbox', 'sample_idx'
]

flip_pairs = [[1, 2], [4, 5], [7, 8], [10, 11], [13, 14], [16, 17], [18, 19],
              [20, 21], [23, 24], [25, 40], [26, 41], [27, 42], [28, 43],
              [29, 44], [30, 45], [31, 46], [32, 47], [33, 48], [34, 49],
              [35, 50], [36, 51], [37, 52], [38, 53], [39, 54], [57, 56],
              [59, 58], [60, 63], [61, 64], [62, 65], [66, 71], [67, 72],
              [68, 73], [69, 74], [70, 75], [81, 80], [82, 79], [83, 78],
              [84, 77], [85, 76], [101, 98], [102, 97], [103, 96], [104, 95],
              [105, 100], [106, 99], [145, 144], [158, 155], [159, 156],
              [160, 157], [165, 162], [166, 163], [167, 164], [169, 168],
              [171, 170], [172, 175], [173, 176], [174, 177], [179, 180],
              [181, 182], [183, 184], [188, 189]]


dataset_type = 'HybrIKHumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

keypoints_maps = [
    dict(
        keypoints=[
            'keypoints3d17',
            'keypoints3d17_vis',
            'keypoints3d17_relative',
        ],
        keypoints_index=h36m_idxs),
    dict(
        keypoints=['keypoints3d', 'keypoints3d_vis', 'keypoints3d_relative'],
        keypoints_index=hybrik29_idxs),
]

class MixDataset(data.Dataset):
    CLASSES = ['person']
    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    num_joints = 24
    bbox_3d_shape = (2000, 2000, 2000)
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    joints_name_24 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb'             # 23
    )
    data_domain = set([
        'dataset_name',
        'type',
        'target_theta',
        'target_theta_weight',
        'target_beta',
        'target_smpl_weight',
        'target_uvd_29',
        'target_xyz_24',
        'target_weight_24',
        'target_weight_29',
        'target_xyz_17',
        'target_weight_17',
        'trans_inv',
        'intrinsic_param',
        'joint_root',
        'target_twist',
        'target_twist_weight',
        'depth_factor'
    ])

    def __init__(self,
                 cfg,
                 train=True):
        self._train = train
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE

        if train:
            self.db0 = H36mSMPL(
                cfg=cfg,
                ann_file=cfg.DATASET.SET_LIST[0].TRAIN_SET,
                train=True,
                lazy_import=True)
            self.db1 = Mscoco(
                cfg=cfg,
                ann_file=f'person_keypoints_{cfg.DATASET.SET_LIST[1].TRAIN_SET}.json',
                train=True,
                lazy_import=True)
            self.db2 = HP3D(
                cfg=cfg,
                ann_file=cfg.DATASET.SET_LIST[2].TRAIN_SET,
                train=True,
                lazy_import=True)

            self._subsets = [self.db0, self.db1, self.db2]
            self._2d_length = len(self.db1)
            self._3d_length = len(self.db0) + len(self.db2)
        else:
            self.db0 = H36mSMPL(
                cfg=cfg,
                ann_file=cfg.DATASET.SET_LIST[0].TEST_SET,
                train=train)

            self._subsets = [self.db0]

        self._subset_size = [len(item) for item in self._subsets]
        self._db0_size = len(self.db0)

        if train:
            self.max_db_data_num = max(self._subset_size)
            self.tot_size = 2 * max(self._subset_size)
            self.partition = [0.4, 0.5, 0.1]
        else:
            self.tot_size = self._db0_size
            self.partition = [1]

        self.cumulative_sizes = self.cumsum(self.partition)

        self.joint_pairs_24 = self.db0.joint_pairs_24
        self.joint_pairs_17 = self.db0.joint_pairs_17
        self.root_idx_17 = self.db0.root_idx_17
        self.root_idx_smpl = self.db0.root_idx_smpl
        self.evaluate_xyz_17 = self.db0.evaluate_xyz_17
        self.evaluate_uvd_24 = self.db0.evaluate_uvd_24
        self.evaluate_xyz_24 = self.db0.evaluate_xyz_24

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __len__(self):
        return self.tot_size

    def __getitem__(self, idx):
        assert idx >= 0
        if self._train:
            p = random.uniform(0, 1)

            dataset_idx = bisect.bisect_right(self.cumulative_sizes, p)

            _db_len = self._subset_size[dataset_idx]

            # last batch: random sampling
            if idx >= _db_len * (self.tot_size // _db_len):
                sample_idx = random.randint(0, _db_len - 1)
            else:  # before last batch: use modular
                sample_idx = idx % _db_len
        else:
            dataset_idx = 0
            sample_idx = idx

        img, target, img_id, bbox = self._subsets[dataset_idx][sample_idx]

        if dataset_idx > 0:
            # COCO, 3DHP
            label_jts_origin = target.pop('target')
            label_jts_mask_origin = target.pop('target_weight')

            label_uvd_29 = torch.zeros(29, 3)
            label_xyz_24 = torch.zeros(24, 3)
            label_uvd_29_mask = torch.zeros(29, 3)
            label_xyz_17 = torch.zeros(17, 3)
            label_xyz_17_mask = torch.zeros(17, 3)

            if dataset_idx == 1:
                # COCO
                assert label_jts_origin.dim() == 1 and label_jts_origin.shape[0] == 17 * 2, label_jts_origin.shape

                label_jts_origin = label_jts_origin.reshape(17, 2)
                label_jts_mask_origin = label_jts_mask_origin.reshape(17, 2)

                for i in range(s_smpl24_jt_num):
                    id1 = i
                    id2 = s_coco_2_smpl_jt[i]
                    if id2 >= 0:
                        label_uvd_29[id1, :2] = label_jts_origin[id2, :2].clone()
                        label_uvd_29_mask[id1, :2] = label_jts_mask_origin[id2, :2].clone()
            elif dataset_idx == 2:
                # 3DHP
                assert label_jts_origin.dim() == 1 and label_jts_origin.shape[0] == 28 * 3, label_jts_origin.shape

                label_jts_origin = label_jts_origin.reshape(28, 3)
                label_jts_mask_origin = label_jts_mask_origin.reshape(28, 3)

                for i in range(s_smpl24_jt_num):
                    id1 = i
                    id2 = s_3dhp_2_smpl_jt[i]
                    if id2 >= 0:
                        label_uvd_29[id1, :3] = label_jts_origin[id2, :3].clone()
                        label_uvd_29_mask[id1, :3] = label_jts_mask_origin[id2, :3].clone()

            label_uvd_29 = label_uvd_29.reshape(-1)
            label_xyz_24 = label_xyz_24.reshape(-1)
            label_uvd_24_mask = label_uvd_29_mask[:24, :].reshape(-1)
            label_uvd_29_mask = label_uvd_29_mask.reshape(-1)
            label_xyz_17 = label_xyz_17.reshape(-1)
            label_xyz_17_mask = label_xyz_17_mask.reshape(-1)

            target['target_uvd_29'] = label_uvd_29
            target['target_xyz_24'] = label_xyz_24
            target['target_weight_24'] = label_uvd_24_mask
            target['target_weight_29'] = label_uvd_29_mask
            target['target_xyz_17'] = label_xyz_17
            target['target_weight_17'] = label_xyz_17_mask
            target['target_theta'] = torch.zeros(24 * 4)
            target['target_beta'] = torch.zeros(10)
            target['target_smpl_weight'] = torch.zeros(1)
            target['target_theta_weight'] = torch.zeros(24 * 4)
            target['target_twist'] = torch.zeros(23, 2)
            target['target_twist_weight'] = torch.zeros(23, 2)
        else:
            assert set(target.keys()).issubset(self.data_domain), (set(target.keys()) - self.data_domain, self.data_domain - set(target.keys()),)
        target.pop('type')
        target.pop('dataset_name')
        return img, target, img_id, bbox

class MixDataset_add3dpw(data.Dataset):
    CLASSES = ['person']
    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    num_joints = 24
    bbox_3d_shape = (2000, 2000, 2000)
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    joints_name_24 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb'             # 23
    )
    data_domain = set([
        'dataset_name',
        'type',
        'target_theta',
        'target_theta_weight',
        'target_beta',
        'target_smpl_weight',
        'target_uvd_29',
        'target_xyz_24',
        'target_weight_24',
        'target_weight_29',
        'target_xyz_17',
        'target_weight_17',
        'trans_inv',
        'intrinsic_param',
        'joint_root',
        'target_twist',
        'target_twist_weight',
        'depth_factor'
    ])

    def __init__(self,
                 cfg,
                 train=True):
        self._train = train
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE

        if train:

            self.db0 = H36mSMPL(
                cfg=cfg,
                ann_file=cfg.DATASET.SET_LIST[0].TRAIN_SET,
                train=True,
                lazy_import=True)
            self.db1 = Mscoco(
                cfg=cfg,
                ann_file=f'person_keypoints_{cfg.DATASET.SET_LIST[1].TRAIN_SET}.json',
                train=True,
                lazy_import=True)
            self.db2 = HP3D(
                cfg=cfg,
                ann_file=cfg.DATASET.SET_LIST[2].TRAIN_SET,
                train=True,
                lazy_import=True)
            self.db3 = PW3D(
                cfg=cfg,
                ann_file=cfg.DATASET.SET_LIST[3].TRAIN_SET+'.json',
                train=True,
                lazy_import=True)


            self._subsets = [self.db0, self.db1, self.db2, self.db3]
            self._2d_length = len(self.db1)
            self._3d_length = len(self.db0) + len(self.db2) + len(self.db3)
        else:
            self.db0 = H36mSMPL(
                cfg=cfg,
                ann_file=cfg.DATASET.SET_LIST[0].TEST_SET,
                train=train)

            self._subsets = [self.db0]

        self._subset_size = [len(item) for item in self._subsets]
        self._db0_size = len(self.db0)

        if train:
            self.max_db_data_num = max(self._subset_size)
            self.tot_size = 2 * max(self._subset_size)
            # self.partition = [0.4, 0.5, 0.1]
            self.partition = [0.2, 0.5, 0.1, 0.2]
        else:
            self.tot_size = self._db0_size
            self.partition = [1]

        self.cumulative_sizes = self.cumsum(self.partition)
        self.joint_pairs_24 = self.db0.joint_pairs_24
        self.joint_pairs_17 = self.db0.joint_pairs_17
        self.root_idx_17 = self.db0.root_idx_17
        self.root_idx_smpl = self.db0.root_idx_smpl
        self.evaluate_xyz_17 = self.db0.evaluate_xyz_17
        self.evaluate_uvd_24 = self.db0.evaluate_uvd_24
        self.evaluate_xyz_24 = self.db0.evaluate_xyz_24

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __len__(self):
        return self.tot_size

    def __getitem__(self, idx):
        assert idx >= 0
        if self._train:
            p = random.uniform(0, 1)

            dataset_idx = bisect.bisect_right(self.cumulative_sizes, p)

            _db_len = self._subset_size[dataset_idx]

            # last batch: random sampling
            if idx >= _db_len * (self.tot_size // _db_len):
                sample_idx = random.randint(0, _db_len - 1)
            else:  # before last batch: use modular
                sample_idx = idx % _db_len
        else:
            dataset_idx = 0
            sample_idx = idx

        img, target, img_id, bbox = self._subsets[dataset_idx][sample_idx]

        if dataset_idx > 0 and dataset_idx < 3:
            # COCO, 3DHP
            label_jts_origin = target.pop('target')
            label_jts_mask_origin = target.pop('target_weight')

            label_uvd_29 = torch.zeros(29, 3)
            label_xyz_24 = torch.zeros(24, 3)
            label_uvd_29_mask = torch.zeros(29, 3)
            label_xyz_17 = torch.zeros(17, 3)
            label_xyz_17_mask = torch.zeros(17, 3)

            if dataset_idx == 1:
                # COCO
                assert label_jts_origin.dim() == 1 and label_jts_origin.shape[0] == 17 * 2, label_jts_origin.shape

                label_jts_origin = label_jts_origin.reshape(17, 2)
                label_jts_mask_origin = label_jts_mask_origin.reshape(17, 2)

                for i in range(s_smpl24_jt_num):
                    id1 = i
                    id2 = s_coco_2_smpl_jt[i]
                    if id2 >= 0:
                        label_uvd_29[id1, :2] = label_jts_origin[id2, :2].clone()
                        label_uvd_29_mask[id1, :2] = label_jts_mask_origin[id2, :2].clone()
            elif dataset_idx == 2:
                # 3DHP
                assert label_jts_origin.dim() == 1 and label_jts_origin.shape[0] == 28 * 3, label_jts_origin.shape

                label_jts_origin = label_jts_origin.reshape(28, 3)
                label_jts_mask_origin = label_jts_mask_origin.reshape(28, 3)

                for i in range(s_smpl24_jt_num):
                    id1 = i
                    id2 = s_3dhp_2_smpl_jt[i]
                    if id2 >= 0:
                        label_uvd_29[id1, :3] = label_jts_origin[id2, :3].clone()
                        label_uvd_29_mask[id1, :3] = label_jts_mask_origin[id2, :3].clone()

            label_uvd_29 = label_uvd_29.reshape(-1)
            label_xyz_24 = label_xyz_24.reshape(-1)
            label_uvd_24_mask = label_uvd_29_mask[:24, :].reshape(-1)
            label_uvd_29_mask = label_uvd_29_mask.reshape(-1)
            label_xyz_17 = label_xyz_17.reshape(-1)
            label_xyz_17_mask = label_xyz_17_mask.reshape(-1)

            target['target_uvd_29'] = label_uvd_29
            target['target_xyz_24'] = label_xyz_24
            target['target_weight_24'] = label_uvd_24_mask
            target['target_weight_29'] = label_uvd_29_mask
            target['target_xyz_17'] = label_xyz_17
            target['target_weight_17'] = label_xyz_17_mask
            target['target_theta'] = torch.zeros(24 * 4)
            target['target_beta'] = torch.zeros(10)
            target['target_smpl_weight'] = torch.zeros(1)
            target['target_theta_weight'] = torch.zeros(24 * 4)
            target['target_twist'] = torch.zeros(23, 2)
            target['target_twist_weight'] = torch.zeros(23, 2)
        else:
            assert set(target.keys()).issubset(self.data_domain), (set(target.keys()) - self.data_domain, self.data_domain - set(target.keys()),)
        target.pop('type')
        target.pop('dataset_name')

        # img = 0
        # # target = 0
        # target = dict()
        # target['target_uvd_29'] = torch.zeros(24 * 4)
        # target['target_xyz_24'] = torch.zeros(24 * 4)
        # target['target_weight_24'] = torch.zeros(24 * 4)
        # target['target_weight_29'] = torch.zeros(24 * 4)
        # target['target_xyz_17'] = torch.zeros(24 * 4)
        # target['target_weight_17'] = torch.zeros(24 * 4)
        # target['target_theta'] = torch.zeros(24 * 4)
        # target['target_beta'] = torch.zeros(10)
        # target['target_smpl_weight'] = torch.zeros(1)
        # target['target_theta_weight'] = torch.zeros(24 * 4)
        # target['target_twist'] = torch.zeros(23, 2)
        # target['target_twist_weight'] = torch.zeros(23, 2)
        #
        # img_id = 0
        # bbox = 0
        # img_path = '../HybrIK_dataset/pw3d/imageFiles/courtyard_laceShoe_00/image_00134.jpg'
        # img1 = cv2.imread(img_path)[:, :, ::-1]



        return img, target, img_id, bbox

class MixDataset_MMStyle(Dataset):
    """Mixed Dataset.

    Args:
        config (list): the list of different datasets.
        partition (list): the ratio of datasets in each batch.
        num_data (int | None, optional): if num_data is not None, the number
            of iterations is set to this fixed value. Otherwise, the number of
            iterations is set to the maximum size of each single dataset.
            Default: None.
    """

    # def __init__(self,
    #              configs: list,
    #              partition: list,
    #              num_data: Optional[Union[int, None]] = None):
    def __init__(self,
                 cfg,
                 train=True):
        """Load data from multiple datasets."""
        assert min(cfg.DATASET.PARTITION) >= 0
        # assert min(partition) >= 0
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='RandomDPG', dpg_prob=0.9),
            dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.3),
            dict(type='RandomOcclusion', occlusion_prob=0.9),
            dict(type='HybrIKRandomFlip', flip_prob=0.5, flip_pairs=flip_pairs),
            dict(type='NewKeypointsSelection', maps=keypoints_maps),
            dict(type='HybrIKAffine', img_res=256),
            dict(type='GenerateHybrIKTarget', img_res=256, test_mode=False),
            dict(type='RandomChannelNoise', noise_factor=0.4),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=data_keys),
            dict(
                type='Collect',
                keys=['img', *data_keys],
                meta_keys=['center', 'scale', 'rotation', 'image_path'])
        ]
        # train_pipeline = [
        #     dict(type='HybrikLoadimg'),
        #     dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.3),
        #     dict(type='RandomOcclusion', occlusion_prob=0.9),
        #     dict(type='HybrIKRandomFlip', flip_prob=0.5, flip_pairs=flip_pairs),
        #     dict(type='NewKeypointsSelection', maps=keypoints_maps),
        #     dict(type='HybrIKAffine', img_res=256),
        #     dict(type='GenerateHybrIKTarget', img_res=256, test_mode=False),
        #     dict(type='HybrikImgProcess', color_factor=0.2),
        #     dict(type='ToTensor', keys=data_keys),
        #     dict(
        #         type='Collect',
        #         keys=['img', *data_keys],
        #         meta_keys=['center', 'scale', 'rotation', 'image_path'])
        # ]
        # train_pipeline = [
        #     dict(type='HybrikLoadimg'),
        #     dict(type='HybrikPipeline'),
        #     dict(
        #         type='Collect',
        #         keys=['img', *data_keys],
        #         meta_keys=['center', 'scale', 'rotation', 'image_path'])
        # ]
        dataset_config = cfg.DATASET.configs

        for i in range(len(dataset_config)):
            dataset_config[i]['pipeline'] = train_pipeline
        # dataset_config[0]['pipeline'] = train_pipeline
        # dataset_config[1]['pipeline'] = train_pipeline
        # dataset_config[2]['pipeline'] = train_pipeline

        datasets = [build_dataset(cfg_) for cfg_ in dataset_config]
        self.dataset = ConcatDataset(datasets)

        self.length = max(len(ds) for ds in datasets)

        weights = [
            np.ones(len(ds)) * p / len(ds)
            for (p, ds) in zip(cfg.DATASET.PARTITION, datasets)
        ]
        weights = np.concatenate(weights, axis=0)
        self.sampler = WeightedRandomSampler(weights, 1)

    def __len__(self):
        """Get the size of the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Given index, sample the data from multiple datasets with the given
        proportion."""
        idx_new = list(self.sampler)[0]
        return self.dataset[idx_new]

class MixDataset_Our(data.Dataset):
    CLASSES = ['person']
    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    num_joints = 24
    bbox_3d_shape = (2000, 2000, 2000)
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    joints_name_24 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb'             # 23
    )
    data_domain = set([
        'dataset_name',
        'type',
        'target_theta',
        'target_theta_weight',
        'target_beta',
        'target_smpl_weight',
        'target_uvd_29',
        'target_xyz_24',
        'target_weight_24',
        'target_weight_29',
        'target_xyz_17',
        'target_weight_17',
        'trans_inv',
        'intrinsic_param',
        'joint_root',
        'target_twist',
        'target_twist_weight',
        'depth_factor'
    ])

    def __init__(self,
                 cfg,
                 train=True):
        self._train = train
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        assert min(cfg.DATASET.PARTITION) >= 0

        if train:

            if len(cfg.DATASET.SET_LIST) == 3:
                self.db0 = H36mSMPL(
                    cfg=cfg,
                    ann_file=cfg.DATASET.SET_LIST[0].TRAIN_SET,
                    train=True,
                    dpg=True,
                    lazy_import=True)
                self.db1 = Mscoco(
                    cfg=cfg,
                    ann_file=f'person_keypoints_{cfg.DATASET.SET_LIST[1].TRAIN_SET}.json',
                    train=True,
                    dpg=True,
                    lazy_import=True)
                self.db2 = HP3D(
                    cfg=cfg,
                    ann_file=cfg.DATASET.SET_LIST[2].TRAIN_SET,
                    train=True,
                    dpg=True,
                    lazy_import=True)
                subsets = [self.db0, self.db1, self.db2]
            else:
                self.db0 = H36mSMPL(
                    cfg=cfg,
                    ann_file=cfg.DATASET.SET_LIST[0].TRAIN_SET,
                    train=True,
                    dpg=True,
                    lazy_import=True)
                self.db1 = Mscoco(
                    cfg=cfg,
                    ann_file=f'person_keypoints_{cfg.DATASET.SET_LIST[1].TRAIN_SET}.json',
                    train=True,
                    dpg=True,
                    lazy_import=True)
                self.db2 = HP3D(
                    cfg=cfg,
                    ann_file=cfg.DATASET.SET_LIST[2].TRAIN_SET,
                    train=True,
                    dpg=True,
                    lazy_import=True)
                self.db3 = PW3D(
                    cfg=cfg,
                    ann_file=cfg.DATASET.SET_LIST[3].TRAIN_SET+'.json',
                    train=True,
                    dpg=True,
                    lazy_import=True)
                subsets = [self.db0, self.db1, self.db2, self.db3]


            self.dataset = ConcatDataset(subsets)

            self.length = max(len(ds) for ds in subsets)

            weights = [
                np.ones(len(ds)) * p / len(ds)
                for (p, ds) in zip(cfg.DATASET.PARTITION, subsets)
            ]
            weights = np.concatenate(weights, axis=0)
            self.sampler = WeightedRandomSampler(weights, 1)

        else:
            self.db0 = H36mSMPL(
                cfg=cfg,
                ann_file=cfg.DATASET.SET_LIST[0].TEST_SET,
                train=train)

            self._subsets = [self.db0]

    def __len__(self):
        """Get the size of the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Given index, sample the data from multiple datasets with the given
        proportion."""
        idx_new = list(self.sampler)[0]

        img, target, img_id, bbox = self.dataset[idx_new]

        if target['dataset_name'] == 'coco':
            # COCO
            label_jts_origin = target.pop('target')
            label_jts_mask_origin = target.pop('target_weight')

            label_uvd_29 = torch.zeros(29, 3)
            label_xyz_24 = torch.zeros(24, 3)
            label_uvd_29_mask = torch.zeros(29, 3)
            label_xyz_17 = torch.zeros(17, 3)
            label_xyz_17_mask = torch.zeros(17, 3)

            assert label_jts_origin.dim() == 1 and label_jts_origin.shape[0] == 17 * 2, label_jts_origin.shape

            label_jts_origin = label_jts_origin.reshape(17, 2)
            label_jts_mask_origin = label_jts_mask_origin.reshape(17, 2)

            for i in range(s_smpl24_jt_num):
                id1 = i
                id2 = s_coco_2_smpl_jt[i]
                if id2 >= 0:
                    label_uvd_29[id1, :2] = label_jts_origin[id2, :2].clone()
                    label_uvd_29_mask[id1, :2] = label_jts_mask_origin[id2, :2].clone()
            label_uvd_29 = label_uvd_29.reshape(-1)
            label_xyz_24 = label_xyz_24.reshape(-1)
            label_uvd_24_mask = label_uvd_29_mask[:24, :].reshape(-1)
            label_uvd_29_mask = label_uvd_29_mask.reshape(-1)
            label_xyz_17 = label_xyz_17.reshape(-1)
            label_xyz_17_mask = label_xyz_17_mask.reshape(-1)

            target['target_uvd_29'] = label_uvd_29
            target['target_xyz_24'] = label_xyz_24
            target['target_weight_24'] = label_uvd_24_mask
            target['target_weight_29'] = label_uvd_29_mask
            target['target_xyz_17'] = label_xyz_17
            target['target_weight_17'] = label_xyz_17_mask
            target['target_theta'] = torch.zeros(24 * 4)
            target['target_beta'] = torch.zeros(10)
            target['target_smpl_weight'] = torch.zeros(1)
            target['target_theta_weight'] = torch.zeros(24 * 4)
            target['target_twist'] = torch.zeros(23, 2)
            target['target_twist_weight'] = torch.zeros(23, 2)

        elif target['dataset_name'] == 'hp3d':
            # 3DHP
            label_jts_origin = target.pop('target')
            label_jts_mask_origin = target.pop('target_weight')

            label_uvd_29 = torch.zeros(29, 3)
            label_xyz_24 = torch.zeros(24, 3)
            label_uvd_29_mask = torch.zeros(29, 3)
            label_xyz_17 = torch.zeros(17, 3)
            label_xyz_17_mask = torch.zeros(17, 3)

            assert label_jts_origin.dim() == 1 and label_jts_origin.shape[0] == 28 * 3, label_jts_origin.shape

            label_jts_origin = label_jts_origin.reshape(28, 3)
            label_jts_mask_origin = label_jts_mask_origin.reshape(28, 3)

            for i in range(s_smpl24_jt_num):
                id1 = i
                id2 = s_3dhp_2_smpl_jt[i]
                if id2 >= 0:
                    label_uvd_29[id1, :3] = label_jts_origin[id2, :3].clone()
                    label_uvd_29_mask[id1, :3] = label_jts_mask_origin[id2, :3].clone()

            label_uvd_29 = label_uvd_29.reshape(-1)
            label_xyz_24 = label_xyz_24.reshape(-1)
            label_uvd_24_mask = label_uvd_29_mask[:24, :].reshape(-1)
            label_uvd_29_mask = label_uvd_29_mask.reshape(-1)
            label_xyz_17 = label_xyz_17.reshape(-1)
            label_xyz_17_mask = label_xyz_17_mask.reshape(-1)

            target['target_uvd_29'] = label_uvd_29
            target['target_xyz_24'] = label_xyz_24
            target['target_weight_24'] = label_uvd_24_mask
            target['target_weight_29'] = label_uvd_29_mask
            target['target_xyz_17'] = label_xyz_17
            target['target_weight_17'] = label_xyz_17_mask
            target['target_theta'] = torch.zeros(24 * 4)
            target['target_beta'] = torch.zeros(10)
            target['target_smpl_weight'] = torch.zeros(1)
            target['target_theta_weight'] = torch.zeros(24 * 4)
            target['target_twist'] = torch.zeros(23, 2)
            target['target_twist_weight'] = torch.zeros(23, 2)
        else:
            assert set(target.keys()).issubset(self.data_domain), (set(target.keys()) - self.data_domain, self.data_domain - set(target.keys()),)
        target.pop('type')
        target.pop('dataset_name')

        return img, target, img_id, bbox

