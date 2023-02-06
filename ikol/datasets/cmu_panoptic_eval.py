"""Human3.6M dataset."""
import copy
import json
import os
import pickle
import cv2
import numpy as np
import torch
import torch.utils.data as data
from ikol.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from ikol.utils.pose_utils import cam2pixel_matrix, pixel2cam_matrix, cam2pixel, pixel2cam, reconstruction_error
from ikol.utils.presets import SimpleTransform3DSMPL
import matplotlib.pyplot as plt
import mmcv

################ CMU annotation order ################
# 0: Neck
# 1: Nose
# 2: BodyCenter (center of hips)
# 3: lShoulder
# 4: lElbow
# 5: lWrist,
# 6: lHip
# 7: lKnee
# 8: lAnkle
# 9: rShoulder
# 10: rElbow
# 11: rWrist
# 12: rHip
# 13: rKnee
# 14: rAnkle
# 15: lEye
# 16: lEar
# 17: rEye
# 18: rEar

# J24_to_J15 = [12, 13, 14, 9, 10, 11,  # 5s
#               3, 4, 5, 8, 7, 6,  # 11
#               2, 1, 0
#               ]

####  J24  0rAnkle  1rKnee  2rHip   3lHip  4lKnee  5lAnkle   6rWrist  7rElbow  8rShoulder  9lShoulder  10lElbow  11lWrist  12Neck  13Nose  14BodyCenter


# self.J24_TO_H36M = np.array([14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6])

####  H36M  0 BodyCenter/Pelvis  1 lHip 2 lKnee 3lAnkle 4rHip 5rKnee 6rAnkle 7unkonwn 8Neck 9unkonwn 10unkonwn 11lShoulder 12lElbow 13lWrist  14rShoulder 15rElbow 16rWrist  Right!!!!





green_frames = ['160422_haggling1-00_16_00002945.jpg',
'160422_haggling1-00_16_00002946.jpg',
'160422_haggling1-00_16_00002947.jpg',
'160422_haggling1-00_16_00002948.jpg',
'160422_haggling1-00_16_00002949.jpg',
'160422_haggling1-00_16_00002950.jpg',
'160422_haggling1-00_16_00002951.jpg',
'160422_haggling1-00_16_00002952.jpg',
'160422_haggling1-00_16_00002953.jpg',
'160422_haggling1-00_16_00002954.jpg',
'160422_haggling1-00_30_00001402.jpg',
'160422_haggling1-00_30_00001403.jpg',
'160422_haggling1-00_30_00001404.jpg',
'160422_haggling1-00_30_00001405.jpg',
'160422_haggling1-00_30_00001406.jpg',
'160422_haggling1-00_30_00001407.jpg',
'160422_haggling1-00_30_00001408.jpg',
'160422_haggling1-00_30_00001409.jpg',
'160422_haggling1-00_30_00001410.jpg',
'160422_haggling1-00_30_00001411.jpg',
'160422_haggling1-00_30_00001412.jpg',
'160422_haggling1-00_30_00001414.jpg']

J24_to_J15 = np.array([12, 13, 14, 9, 10, 11,  # 5s
                            3, 4, 5, 8, 7, 6,  # 11
                            2, 1, 0])
scale = (832, 512)




def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)



def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord


def WorldprojectPoints(X, K, R, t, Kd):
    """ Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].

    Roughly, x = K*(R*X + t) + distortion

    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """

    # x = np.asarray(R*X + t)
    x = np.asarray(np.dot(R, X) + t)

    x[0:2, :] = x[0:2, :] / x[2, :]

    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
                r + 2 * x[0, :] * x[0, :])
    x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
                r + 2 * x[1, :] * x[1, :])

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    return x

def CamprojectPoints(X, K, Kd):
    """ Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].

    Roughly, x = K*(R*X + t) + distortion

    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """

    # x = np.asarray(R*X + t)
    # x = np.asarray(np.dot(R, X) + t)
    x = X.copy()

    x[0:2, :] = x[0:2, :] / x[2, :]

    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
                r + 2 * x[0, :] * x[0, :])
    x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
                r + 2 * x[1, :] * x[1, :])

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    return x


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord


def distortPoints(undistortedPoints, k, d):
    undistorted = np.float32(undistortedPoints[:, np.newaxis, :])

    kInv = np.linalg.inv(k)

    for i in range(len(undistorted)):
        srcv = np.array([undistorted[i][0][0], undistorted[i][0][1], 1])
        dstv = kInv.dot(srcv)
        undistorted[i][0][0] = dstv[0]
        undistorted[i][0][1] = dstv[1]

    distorted = cv2.fisheye.distortPoints(undistorted, k, d)
    return distorted


class CMU_Panoptic_eval(data.Dataset):
    # EVAL_JOINTS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    def __init__(self, cfg, train=True, root='./data/datasets/', split='test', dpg=False,):

        self.data_folder = os.path.join(root, 'cmu_panoptic/')
        self.min_pts_required = 5
        self.split = split
        self.J24_TO_H36M = np.array([14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6])
        self.H36M_TO_LSP = self.J24_TO_H36M[np.array([6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10])]
        # self.annots_folder = os.path.join(self.data_folder, 'panoptic_annot')
        # self.load_annots()
        self.annots_folder = os.path.join(self.data_folder, 'annotation_our')
        self.load_annots_our()
        self.image_folder = os.path.join(self.data_folder, 'images/')

        # for green_frame in green_frames:
        #     del self.annots[green_frame]

        # self.EVAL_JOINTS_17 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        self.EVAL_JOINTS_17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]


        # self._items, self._labels = self._lazy_load_json()
        self.root_inds = 0  # [constants.SMPL_ALL_54['R_Hip'], constants.SMPL_ALL_54['L_Hip']]
        self.root_idx_17 = 0
        self._train = train
        self._scale_factor = cfg.DATASET.SCALE_FACTOR
        self._color_factor = cfg.DATASET.COLOR_FACTOR
        self._rot = cfg.DATASET.ROT_FACTOR
        self._input_size = cfg.MODEL.IMAGE_SIZE
        self._output_size = cfg.MODEL.HEATMAP_SIZE
        self._crop = cfg.MODEL.EXTRA.CROP
        self._sigma = cfg.MODEL.EXTRA.SIGMA
        self._depth_dim = cfg.MODEL.EXTRA.DEPTH_DIM
        self.bbox_3d_shape = (200, 200, 200)
        self._loss_type = cfg.LOSS['TYPE']
        self._dpg = dpg
        if cfg.MODEL.EXTRA.PRESET == 'simple_smpl_3d':
            self.transformation = SimpleTransform3DSMPL(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=False,
                input_size=self._input_size,
                output_size=self._output_size,
                depth_dim=self._depth_dim,
                bbox_3d_shape=self.bbox_3d_shape,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type, two_d=True)


    def load_annots_our(self):

        ann_file = os.path.join(self.annots_folder, 'annotation.pkl')
        with open(ann_file, 'rb') as f:
            annots = pickle.load(f)

        self._labels = annots

    def determine_visible_person(self, kp2ds, width, height):
        visible_person_id,kp2d_vis = [],[]
        for person_id,kp2d in enumerate(kp2ds):
            visible_kps_mask = np.logical_and(np.logical_and(0<kp2d[:,0],kp2d[:,0]<width),np.logical_and(0<kp2d[:,1],kp2d[:,1]<height,kp2d[:,2]>0))
            if visible_kps_mask.sum()>1:
                visible_person_id.append(person_id)
                kp2d_vis.append(np.concatenate([kp2d[:,:2], visible_kps_mask[:,None]],1))
        return np.array(visible_person_id), np.array(kp2d_vis)

    def __getitem__(self, idx):

        # get image id
        self.image_folder = os.path.join(self.data_folder, 'images/')

        img_path = os.path.join(self.image_folder, self._labels[idx]['img_path'])
        img_id = int(self._labels[idx]['img_id'])

        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        img = cv2.imread(img_path)[:,:,::-1]

        # for i in range(5):
        #     torch.cuda.synchronize()
        #     start_time = time.perf_counter()
        #     img = scipy.misc.imread(img_path, mode='RGB')
        #     torch.cuda.synchronize()
        #     elapsed = time.perf_counter() - start_time
        #     print(elapsed)


        # import cv2
        # for i in range(5):
        #     torch.cuda.synchronize()
        #     start_time = time.perf_counter()
        #     img1 = cv2.imread(img_path)
        #     torch.cuda.synchronize()
        #     elapsed = time.perf_counter() - start_time
        #     print(elapsed)
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(img1)
        # plt.show()

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox

    def __len__(self):
        return len(self._labels)


    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        hp3d_joint_pairs = ((8, 13), (9, 14), (10, 15), (11, 16), (12, 17),
                            (18, 23), (19, 24), (20, 25), (21, 26), (22, 27))
        return hp3d_joint_pairs
        # return ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))  # h36m pairs

    def evaluate_xyz_17(self, preds, result_dir):
        print('Evaluation start...')
        gts = self._labels
        assert len(gts) == len(preds)
        sample_num = len(gts)

        pred_save = []
        error = np.zeros((sample_num, 17))  # joint error
        error_pa = np.zeros((sample_num, 17))  # joint error
        error_x = np.zeros((sample_num, 17))  # joint error
        error_y = np.zeros((sample_num, 17))  # joint error
        error_z = np.zeros((sample_num, 17))
        vis_masks = np.zeros((sample_num, 17),dtype=bool) # joint error
        # error for each sequence
        for n in range(sample_num):
            gt = gts[n]
            img_name = gt['img_name']

            # intrinsic_param = gt['intrinsic_param']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            gt_3d_kpt = gt['joint_cam']

            # gt_vis = gt['joint_vis']
            pred_3d_kpt = preds[n]['xyz_17'].copy() * self.bbox_3d_shape[2]

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_17]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_17]

            pred_3d_kpt = np.take(pred_3d_kpt, self.EVAL_JOINTS_17, axis=0)
            gt_3d_kpt = np.take(gt_3d_kpt, self.EVAL_JOINTS_17, axis=0)
            vis_mask = gt_3d_kpt[:, -1] != -2.
            # if self.protocol == 1:
            #     # rigid alignment for PA MPJPE (protocol #1)
            pred_3d_kpt_pa = reconstruction_error(pred_3d_kpt, gt_3d_kpt)
            align = False
            if align:
                pred_3d_kpt = pred_3d_kpt_pa


            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_pa[n] = np.sqrt(np.sum((pred_3d_kpt_pa - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            vis_masks[n] = vis_mask


            # record idx per seq or act
            # seq_id = int(img_name.split('/')[-3][2])
            # seq_idx_dict[seq_id].append(n)
            # act_idx_dict[int(gt['activity_id']) - 1].append(n)

            img_name = gt['img_path']
            # prediction save
            pred_save.append({'img_name': img_name, 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': [float(_) for _ in bbox], 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error[vis_masks]*10.)
        tot_err_pa = np.mean(error_pa[vis_masks]*10.)
        tot_err_x = np.mean(error_x[vis_masks]*10.)
        tot_err_y = np.mean(error_y[vis_masks]*10.)
        tot_err_z = np.mean(error_z[vis_masks]*10.)

        eval_summary = f'PA MPJPE >> tot: {tot_err_pa:2f}; MPJPE >> tot: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'
        print(eval_summary)

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err


class CMU_Panoptic_Preprocess(data.Dataset):

    def __init__(self, train=True, root='./data/datasets/', split='test', dpg=False,):

        self.data_folder = os.path.join(root, 'cmu_panoptic/')
        self.min_pts_required = 5
        self.split = split
        self.J24_TO_H36M = np.array([14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6])
        self.H36M_TO_LSP = self.J24_TO_H36M[np.array([6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10])]
        self.annots_folder = os.path.join(self.data_folder, 'panoptic_annot')
        self.load_annots()
        for green_frame in green_frames:
            del self.annots[green_frame]

        self._items, self._labels = self._lazy_load_json()
        self.root_inds = 0  # [constants.SMPL_ALL_54['R_Hip'], constants.SMPL_ALL_54['L_Hip']]


    def load_annots(self):
        self.annots = {}
        for annots_file_name in os.listdir(self.annots_folder):
            ann_file = os.path.join(self.annots_folder, annots_file_name)
            with open(ann_file, 'rb') as f:
                img_infos = pickle.load(f)
            for img_info in img_infos:
                img_path = img_info['filename'].split('/')
                img_name = img_path[1]+'-'+img_path[-1].replace('.png', '.jpg')
                self.annots[img_name] = {}
                self.annots[img_name] = img_info

    def determine_visible_person(self, kp2ds, width, height):
        visible_person_id,kp2d_vis = [],[]
        for person_id,kp2d in enumerate(kp2ds):
            visible_kps_mask = np.logical_and(np.logical_and(0<kp2d[:,0],kp2d[:,0]<width),np.logical_and(0<kp2d[:,1],kp2d[:,1]<height,kp2d[:,2]>0))
            if visible_kps_mask.sum()>1:
                visible_person_id.append(person_id)
                kp2d_vis.append(np.concatenate([kp2d[:,:2], visible_kps_mask[:,None]],1))
        return np.array(visible_person_id), np.array(kp2d_vis)


    def _lazy_load_json(self):
        """Load all image paths and labels from json annotation files into buffer."""


        items = []
        labels = []
        db = self.annots
        cnt = 0
        for aid in db:
            abs_path = os.path.join('Testset', db[aid]['filename'][10:-4]) + '.jpg'
            act = db[aid]['filename'][10:-4].split('/')[0]
            cam = db[aid]['filename'][10:-4].split('/')[1]
            frame_index = db[aid]['filename'][10:-4].split('/')[2][-8:]
            ####### camera matrix##############
            calibration_path = '../HybrIK_dataset/Panoptic/Testset' + '/' + act + '/calibration_' + act + '.json'
            with open(calibration_path, 'r') as f:
                calibration = json.load(f)
            for j in range(len(calibration['cameras'])):
                if calibration['cameras'][j]['name'] == cam:
                    camera = calibration['cameras'][j]

            ####### pose label##############
                if act == '160422_mafia2':
                    in_path = '../HybrIK_dataset/Panoptic/Testset'  + '/' + act + '/hdPose3d_stage1/body3DScene_' + frame_index + '.json'
                    joint_name = 'joints15'
                else:
                    in_path = '../HybrIK_dataset/Panoptic/Testset'  +'/'+ act + '/hdPose3d_stage1_coco19/body3DScene_' + frame_index + '.json'
                    joint_name = 'joints19'

                with open(in_path, 'r') as f:
                    data = json.load(f)


            K = np.array(camera['K'])
            R = np.array(camera['R'])
            t = np.array(camera['t'])
            distCoef = np.array(camera['distCoef'])

            intrinsic_param = np.zeros([3,4], dtype=np.float32)
            f = np.array([K[0, 0], K[1, 1]], dtype=np.float32)
            c = np.array([K[0, 2], K[1, 2]], dtype=np.float32)
            intrinsic_param[:3,:3] = K
            visible_person_id, kp2ds = self.determine_visible_person(db[aid]['kpts2d'], db[aid]['width'], db[aid]['height'])
            kp3ds = db[aid]['kpts3d'][visible_person_id]
            bboxes = db[aid]['bboxes'][visible_person_id].astype(float)

            for inds, (kp2d, kp3d, bbox) in enumerate(zip(kp2ds, kp3ds, bboxes)):
                kp2d = kp2d[J24_to_J15, :]
                invis_kps = kp2d[:, -1] < 0.1
                kp2d[:, :2] *= 1920. / 832.
                kp2d[invis_kps] = -2.
                kp2d[1] = -2.
                bbox *= 1920. / 832.
                bbox = bbox.round().astype(int)
                invis_3dkps = kp3d[:, -1] < 0.1



                kp3d[invis_3dkps] = -2.
                kp3d = kp3d[self.J24_TO_H36M]

                kp3d_world_original = np.array(data['bodies'][inds][joint_name]).reshape((-1, 4))
                joint_cam_original = world2cam(kp3d_world_original[:, :3], R, t)
                root_cam = joint_cam_original[2]

                if act != '160422_mafia2':
                    kp3d[8,:3] -= np.array([0.0, 0.06, 0.0])  # fix the skeleton misalign

                joint_cam = kp3d[:,:3]
                joint_cam = joint_cam * 100. + root_cam
                joint_img = CamprojectPoints(joint_cam.transpose(), K, distCoef).transpose()
                # joint_img_raw = cam2pixel_matrix(joint_cam, intrinsic_param)
                # joint_img_undist = cv2.undistortPoints(joint_img[:, :2], K, distCoef, None, K).reshape(17, 2)


                reproject = False
                if reproject == True:
                    joint_cam_reproject = pixel2cam(joint_img, f, c)
                    invis_cam = joint_cam[:, -1] == -2.
                    joint_cam_reproject[invis_cam] = -2.
                    # joint_cam = joint_cam_reproject

                joint_img[:, 2] = joint_img[:, 2] - joint_cam[0, 2]
                joint_vis = np.ones((17, 3))


                # skeleton = (
                #     (1, 0), (2, 1), (3, 2),  # 2
                #     (4, 0), (5, 4), (6, 5),  # 5
                #     (7, 0), (8, 7),  # 7
                #     (9, 8), (10, 9),  # 9
                #     (11, 7), (12, 11), (13, 12),  # 12
                #     (14, 7), (15, 14), (16, 15),  # 15
                # )
                # abs_path = os.path.join('../HybrIK_dataset/Panoptic/Testset', db[aid]['filename'][10:-4]) + '.jpg'
                # img = cv2.imread(abs_path)[:, :, ::-1]
                # vis_img = img
                # vis_img = vis_img.copy()
                # vis_kps = np.zeros((3, 17))
                # vis_kps[0, :] = joint_img[:, 0]
                # vis_kps[1, :] = joint_img[:, 1]
                # vis_kps[2, :] = 1
                # vis_img = vis_keypoints(vis_img, vis_kps, skeleton)
                # # vis_img = vis_keypoints(imgs, pred_2d[0,:,:2], skeleton)
                # plt.imshow(vis_img)
                # plt.show()
                # aaa = 1



                items.append(abs_path)
                labels.append({
                    'bbox': bbox,
                    'img_id': cnt,
                    'img_path': abs_path,
                    'img_name': aid,
                    'width': 1920,
                    'height': 1080,
                    'joint_img': joint_img,
                    'joint_vis': joint_vis,
                    'joint_cam': joint_cam,
                    'root_cam': root_cam,
                    'intrinsic_param': intrinsic_param,
                    'distCoef':distCoef,
                    'f': f,
                    'c': c
                })

                cnt += 1

        annotation_path = './data/datasets/cmu_panoptic/annotation_our/annotation.pkl'
        with open(annotation_path, 'wb') as f:
            pickle.dump(labels, f)


        return items, labels


    def load_annots_our(self):

        ann_file = os.path.join(self.annots_folder, 'annotation.pkl')
        with open(ann_file, 'rb') as f:
            annots = pickle.load(f)

        self._labels = annots

    def determine_visible_person(self, kp2ds, width, height):
        visible_person_id,kp2d_vis = [],[]
        for person_id,kp2d in enumerate(kp2ds):
            visible_kps_mask = np.logical_and(np.logical_and(0<kp2d[:,0],kp2d[:,0]<width),np.logical_and(0<kp2d[:,1],kp2d[:,1]<height,kp2d[:,2]>0))
            if visible_kps_mask.sum()>1:
                visible_person_id.append(person_id)
                kp2d_vis.append(np.concatenate([kp2d[:,:2], visible_kps_mask[:,None]],1))
        return np.array(visible_person_id), np.array(kp2d_vis)

    def get_image_info(self, index):
        img_name = self.file_paths[index%len(self.file_paths)]
        imgpath = os.path.join(self.image_folder,img_name)
        image = cv2.imread(imgpath)[:,:,::-1]

        visible_person_id, kp2ds = self.determine_visible_person(self.annots[img_name]['kpts2d'], self.annots[img_name]['width'],self.annots[img_name]['height'])
        kp3ds = self.annots[img_name]['kpts3d'][visible_person_id]
        full_kp2d, kp_3ds, valid_mask_2d, valid_mask_3d = [], [], [], []
        for inds, (kp2d, kp3d) in enumerate(zip(kp2ds, kp3ds)):
            invis_kps = kp2d[:,-1]<0.1
            kp2d *= 1920./832.
            kp2d[invis_kps] = -2.
            kp2d = self.map_kps(kp2d[self.H36M_TO_LSP],maps=self.joint_mapper)
            kp2d[constants.SMPL_ALL_54['Head_top']] = -2.
            full_kp2d.append(kp2d)
            valid_mask_2d.append([True,False,True])
            invis_3dkps = kp3d[:,-1]<0.1
            kp3d = kp3d[:,:3]
            kp3d[invis_3dkps] = -2.
            kp3d = kp3d[self.J24_TO_H36M]
            kp3d[0] -= np.array([0.0,0.06,0.0])#fix the skeleton misalign
            kp_3ds.append(kp3d)
            valid_mask_3d.append([True,False,False,False])

        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape
        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': full_kp2d, 'track_ids': None,\
                'vmask_2d': np.array(valid_mask_2d), 'vmask_3d': np.array(valid_mask_3d),\
                'kp3ds': kp_3ds, 'params': None, 'img_size': image.shape[:2], 'ds': 'cmup'}
        return img_info


    def get_item_single_frame(self,index):
        # valid annotation flags for
        # 0: 2D pose/bounding box(True/False), # detecting all person/front-view person(True/False)
        # 1: 3D pose, 2: subject id, 3: smpl root rot, 4: smpl pose param, 5: smpl shape param
        valid_masks = np.zeros((self.max_person, 6), dtype=np.bool)
        info = self.get_image_info(index)
        scale, rot, flip, color_jitter, syn_occlusion = self._calc_csrfe()
        mp_mode = self._check_mp_mode_()

        img_info = process_image(info['image'], info['kp2ds'], augments=(scale, rot, flip), is_pose2d=info['vmask_2d'][:,0], multiperson=mp_mode)
        if img_info is None:
            return self.resample()
        image, image_wbg, full_kps, offsets = img_info
        centermap, person_centers, full_kp2ds, used_person_inds, valid_masks[:,0], bboxes_hw_norm, heatmap, AE_joints = \
            self.process_kp2ds_bboxes(full_kps, img_shape=image.shape, is_pose2d=info['vmask_2d'][:,0])

        all_person_detected_mask = info['vmask_2d'][0,2]
        subject_ids, valid_masks[:,2] = self.process_suject_ids(info['track_ids'], used_person_inds, valid_mask_ids=info['vmask_2d'][:,1])
        image, dst_image, org_image = self.prepare_image(image, image_wbg, augments=(color_jitter, syn_occlusion))

        # valid mask of 3D pose, smpl root rot, smpl pose param, smpl shape param, global translation
        kp3d, valid_masks[:,1] = self.process_kp3ds(info['kp3ds'], used_person_inds, \
            augments=(rot, flip), valid_mask_kp3ds=info['vmask_3d'][:, 0])
        params, valid_masks[:,3:6] = self.process_smpl_params(info['params'], used_person_inds, \
            augments=(rot, flip), valid_mask_smpl=info['vmask_3d'][:, 1:4])

        input_data = {
            'image': torch.from_numpy(dst_image).float(),
            'image_org': torch.from_numpy(org_image),
            'full_kp2d': torch.from_numpy(full_kp2ds).float(),
            'person_centers':torch.from_numpy(person_centers).float(),
            'subject_ids':torch.from_numpy(subject_ids).long(),
            'centermap': centermap.float(),
            'kp_3d': torch.from_numpy(kp3d).float(),
            'params': torch.from_numpy(params).float(),
            'valid_masks':torch.from_numpy(valid_masks).bool(),
            'offsets': torch.from_numpy(offsets).float(),
            'rot_flip': torch.Tensor([rot, flip]).float(),
            'all_person_detected_mask':torch.Tensor([all_person_detected_mask]).bool(),
            'imgpath': info['imgpath'],
            'data_set': info['ds']}

        # if args().learn_2dpose:
        #     input_data.update({'heatmap':torch.from_numpy(heatmap).float()})
        # if args().learn_AE:
        #     input_data.update({'AE_joints': torch.from_numpy(AE_joints).long()})

        return input_data


    def __getitem__(self, idx):

        # get image id
        self.image_folder = os.path.join(self.data_folder, 'images/')

        img_path = os.path.join(self.image_folder, self._labels[idx]['img_path'])
        img_id = int(self._labels[idx]['img_id'])

        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        img = cv2.imread(img_path)[:,:,::-1]

        # for i in range(5):
        #     torch.cuda.synchronize()
        #     start_time = time.perf_counter()
        #     img = scipy.misc.imread(img_path, mode='RGB')
        #     torch.cuda.synchronize()
        #     elapsed = time.perf_counter() - start_time
        #     print(elapsed)


        # import cv2
        # for i in range(5):
        #     torch.cuda.synchronize()
        #     start_time = time.perf_counter()
        #     img1 = cv2.imread(img_path)
        #     torch.cuda.synchronize()
        #     elapsed = time.perf_counter() - start_time
        #     print(elapsed)
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(img1)
        # plt.show()

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox

    def __len__(self):
        return len(self._labels)


    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        hp3d_joint_pairs = ((8, 13), (9, 14), (10, 15), (11, 16), (12, 17),
                            (18, 23), (19, 24), (20, 25), (21, 26), (22, 27))
        return hp3d_joint_pairs
        # return ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))  # h36m pairs



if __name__ == '__main__':
    # dataset = CMU_Panoptic_eval(train=False)
    dataset = CMU_Panoptic_Preprocess(train=False)
    # test_dataset(dataset)
    print('Done')