import torch
import os
import tqdm
from torch.utils.data import Dataset
from wandb.util import downsample

import articulate as art
import config
from utils.kp_utils import *
from config import device, body_model


class AISTDataset(Dataset):
    """
        AIST dataset. Adapted for different kinds of data.
    """
    def __init__(self, data_dir, kind, split_size=-1, return_mp=True, views=9, load_run_3d=False, down_sample=False):
        super(AISTDataset, self).__init__()
        print('Reading %s dataset "%s"' % (kind, data_dir))
        self.return_mp = return_mp
        self.down_sample = down_sample
        if down_sample:
            split_size *= 2
        dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        dataset2 = None
        self.load_run_3d = load_run_3d
        if load_run_3d:
            assert kind != 'test'
            if kind == 'val':
                dataset2 = torch.load(os.path.join(data_dir, 'val_3d.pt'))
            else:
                d3_1 = torch.load(os.path.join(data_dir, 'train_3d_0-3000.pt'))
                d3_2 = torch.load(os.path.join(data_dir, 'train_3d_3000-6000.pt'))
                d3_3 = torch.load(os.path.join(data_dir, 'train_3d_6000-9000.pt'))
                d3_4 = torch.load(os.path.join(data_dir, 'train_3d_9000-12000.pt'))
                d3_5 = torch.load(os.path.join(data_dir, 'train_3d_12000-15000.pt'))
                dataset2 = {**d3_1, **d3_2, **d3_3, **d3_4, **d3_5}
        j2dcs, j2dc_mps, oriws, accws, poses, trans, Tcws, j3dws, j2dc_mps2, j2dc_mps3, names, j3dr_runs, v3dws, j2dc_mp_origins, s_trans = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        self.j2dcs, self.j2dc_mps, self.oriws, self.accws, self.poses, self.trans, self.Tcws, self.j3dws, self.j2dc_mps2, self.j2dc_mps3, self.names, self.j3dr_runs, self.v3dws, self.j2dc_mp_origins, self.s_trans = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
            for j in range(views):  # jth camera view
                # i = 0
                if dataset['joint2d_mp'][i][j] is None: continue
                Tcw = dataset['cam_T'][i][j]
                # Rrw = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i][:, :3]).transpose(1, 2)
                oriw = dataset['imu_ori'][i]
                accw = dataset['imu_acc'][i]
                j3dw = dataset['joint3d'][i]
                v3dw = (dataset['joint3d'][i][2:] - dataset['joint3d'][i][:-2]) * 30
                v3dw = torch.cat((torch.zeros(1, 3), v3dw[:, 0], torch.zeros(1, 3)), dim=0) / config.vel_scale
                name = dataset['name'][i].replace('cAll', 'c0%d' % (j + 1))
                Kinv = dataset['cam_K'][i][j].inverse()
                j3dc_mp = Tcw.matmul(art.math.append_one(dataset['sync_3d_mp'][i]).unsqueeze(-1)).squeeze(-1)[..., :3]
                j2dc = j3dc_mp / j3dc_mp[..., -1:]
                j2dc[..., :2] = j2dc[..., :2] / (get_bbox_scale(j2dc)).view(-1, 1, 1)
                j2dc = j2dc[..., :2]
                # j2dc = j2dc[:, 1:] - j2dc[:, :1]
                j2dc[:, 24:] = j2dc[:, 24:] - j2dc[:, 23:24]
                j2dc[:, :23] = j2dc[:, :23] - j2dc[:, 23:24]

                j2dc_mp = torch.zeros(len(oriw), 33, 3)
                j2dc_mp[..., :2] = dataset['joint2d_mp'][i][j][..., :2]
                j2dc_mp[..., 0] = j2dc_mp[..., 0] * 1920
                j2dc_mp[..., 1] = j2dc_mp[..., 1] * 1080
                j2dc_mp_origin = j2dc_mp.clone()
                j2dc_mp_origin[..., -1] = dataset['joint2d_mp'][i][j][..., -1]

                j2dc_mp = Kinv.matmul(art.math.append_one(j2dc_mp[..., :2]).unsqueeze(-1)).squeeze(-1)
                j2dc_mp3 = j2dc_mp.clone()
                j2dc_mp[..., :2] = j2dc_mp[..., :2] / (get_bbox_scale(j2dc_mp)).view(-1, 1, 1)
                j2dc_mp2 = j2dc_mp.clone()
                j2dc_mp[:, 1:, :2] = j2dc_mp[:, 1:, :2] - j2dc_mp[:, :1, :2]
                j2dc_mp2[:, 24:, :2] = j2dc_mp2[:, 24:, :2] - j2dc_mp2[:, 23:24, :2]
                j2dc_mp2[:, :23, :2] = j2dc_mp2[:, :23, :2] - j2dc_mp2[:, 23:24, :2]
                j2dc_mp[..., -1] = dataset['joint2d_mp'][i][j][..., -1]
                j2dc_mp2[..., -1] = dataset['joint2d_mp'][i][j][..., -1]
                j2dc_mp3[..., -1] = dataset['joint2d_mp'][i][j][..., -1]
                j2dc_mp = j2dc_mp[:, 1:]

                pose = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i]).view(-1, 24, 3, 3)
                tranc = Tcw.matmul(art.math.append_one(dataset['tran'][i]).unsqueeze(-1)).squeeze(-1)[..., :3]
                s_tran = tranc.clone()
                s_tran[1:] = tranc[:-1]

                # j3dr_run = [None for _ in range(pose.shape[0])]
                j3dr_run = j3dw
                if load_run_3d and name in dataset2:
                    j3dr_run = dataset2[name]

                j2dcs.append(j2dc)
                j2dc_mps.append(j2dc_mp)
                oriws.append(oriw)
                accws.append(accw)
                poses.append(pose)
                j3dws.append(j3dw)
                Tcws.append(Tcw.repeat(len(pose), 1, 1))
                j2dc_mps2.append(j2dc_mp2)
                j2dc_mps3.append(j2dc_mp3)
                names.append(dataset['name'][i].replace('cAll', 'c0%d' % (j + 1)))
                j3dr_runs.append(j3dr_run)
                v3dws.append(v3dw)
                trans.append(tranc)
                s_trans.append(s_tran)
                j2dc_mp_origins.append(j2dc_mp_origin)
                # break
            # break

                if dataset['joint2d_occ'][i][j] is None or len(dataset['joint2d_occ'][i][j]) != len(oriw) or kind == 'test': continue
                j2dc_occ = torch.zeros(len(oriw), 33, 3)
                j2dc_occ[..., :2] = dataset['joint2d_occ'][i][j][..., :2]
                j2dc_occ[..., 0] = j2dc_occ[..., 0] * 1920
                j2dc_occ[..., 1] = j2dc_occ[..., 1] * 1080
                j2dc_mp_origin = j2dc_occ.clone()
                j2dc_mp_origin[..., -1] = dataset['joint2d_occ'][i][j][..., -1]
                
                j2dc_occ = Kinv.matmul(art.math.append_one(j2dc_occ[..., :2]).unsqueeze(-1)).squeeze(-1)
                j2dc_occ3 = j2dc_occ.clone()
                j2dc_occ[..., :2] = j2dc_occ[..., :2] / (get_bbox_scale(j2dc_occ)).view(-1, 1, 1)
                j2dc_occ2 = j2dc_occ.clone()
                j2dc_occ2[:, 24:, :2] = j2dc_occ2[:, 24:, :2] - j2dc_occ2[:, 23:24, :2]
                j2dc_occ2[:, :23, :2] = j2dc_occ2[:, :23, :2] - j2dc_occ2[:, 23:24, :2]
                j2dc_occ[:, 1:, :2] = j2dc_occ[:, 1:, :2] - j2dc_occ[:, :1, :2]
                j2dc_occ[..., -1] = dataset['joint2d_occ'][i][j][..., -1]
                j2dc_occ2[..., -1] = dataset['joint2d_occ'][i][j][..., -1]
                j2dc_occ3[..., -1] = dataset['joint2d_occ'][i][j][..., -1]
                j2dc_occ = j2dc_occ[:, 1:]
                
                name = dataset['name'][i].replace('cAll', 'c0%d' % (j + 1)) + '_occ'
                # j3dr_run = [None for _ in range(pose.shape[0])]
                j3dr_run = j3dw
                if load_run_3d and name in dataset2:
                    j3dr_run = dataset2[name]
                
                j2dcs.append(j2dc)
                j2dc_mps.append(j2dc_occ)
                oriws.append(oriw)
                accws.append(accw)
                poses.append(pose)
                j3dws.append(j3dw)
                Tcws.append(Tcw.repeat(len(pose), 1, 1))
                j2dc_mps2.append(j2dc_occ2)
                j2dc_mps3.append(j2dc_occ3)
                names.append(dataset['name'][i].replace('cAll', 'c0%d' % (j + 1)) + '_occ')
                j3dr_runs.append(j3dr_run)
                v3dws.append(v3dw)
                trans.append(tranc)
                s_trans.append(s_tran)
                j2dc_mp_origins.append(j2dc_mp_origin)

        if split_size > 0:
            for t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14 in \
                    zip(j2dcs, j2dc_mps, oriws, accws, poses, Tcws, j3dws, j2dc_mps2, j3dr_runs, v3dws, trans, j2dc_mp_origins, j2dc_mps3, s_trans):
                if len(t1) < split_size: continue
                m1 = list(t1.split(split_size))
                m1[-1] = t1[-split_size:]
                self.j2dcs.extend(m1)
                m2 = list(t2.split(split_size))
                m2[-1] = t2[-split_size:]
                self.j2dc_mps.extend(m2)
                m3 = list(t3.split(split_size))
                m3[-1] = t3[-split_size:]
                self.oriws.extend(m3)
                m4 = list(t4.split(split_size))
                m4[-1] = t4[-split_size:]
                self.accws.extend(m4)
                m5 = list(t5.split(split_size))
                m5[-1] = t5[-split_size:]
                self.poses.extend(m5)
                m6 = list(t6.split(split_size))
                m6[-1] = t6[-split_size:]
                self.Tcws.extend(m6)
                m7 = list(t7.split(split_size))
                m7[-1] = t7[-split_size:]
                self.j3dws.extend(m7)
                m8 = list(t8.split(split_size))
                m8[-1] = t8[-split_size:]
                self.j2dc_mps2.extend(m8)
                m9 = list(t9.split(split_size))
                m9[-1] = t9[-split_size:]
                self.j3dr_runs.extend(m9)
                m10 = list(t10.split(split_size))
                m10[-1] = t10[-split_size:]
                self.v3dws.extend(m10)
                m11 = list(t11.split(split_size))
                m11[-1] = t11[-split_size:]
                self.trans.extend(m11)
                m12 = list(t12.split(split_size))
                m12[-1] = t12[-split_size:]
                self.j2dc_mp_origins.extend(m12)
                m13 = list(t13.split(split_size))
                m13[-1] = t13[-split_size:]
                self.j2dc_mps3.extend(m13)
                m14 = list(t14.split(split_size))
                m14[-1] = t14[-split_size:]
                self.s_trans.extend(m14)
        else:
            self.j2dcs, self.j2dc_mps, self.oriws, self.accws, self.poses, self.Tcws, self.j3dws, self.j2dc_mps2, self.j3dr_runs, self.v3dws, self.trans, self.j2dc_mp_origins, self.j2dc_mps3, self.s_trans = j2dcs, j2dc_mps, oriws, accws, poses, Tcws, j3dws, j2dc_mps2, j3dr_runs, v3dws, trans, j2dc_mp_origins, j2dc_mps3, s_trans
        if split_size < 0:
            self.names = names

    def __getitem__(self, i):
        item = {}
        item['j2dc'] = self.j2dcs[i].reshape(-1, 33, 2)
        if self.return_mp:
            item['j2dc_mp'] = self.j2dc_mps[i].reshape(-1, 32, 3)
            item['j2dc_mp2'] = self.j2dc_mps2[i].reshape(-1, 33, 3)
            item['j2dc_mp3'] = self.j2dc_mps3[i].reshape(-1, 33, 3)
            # item['j2dc_mp_origin'] = self.j2dc_mp_origins[i].reshape(-1, 33, 3)
        oriw = self.oriws[i].reshape(-1, 6, 3, 3)
        accw = self.accws[i].reshape(-1, 6, 3, 1)
        pose = self.poses[i]
        Tcw = self.Tcws[i]
        j3dw = self.j3dws[i].reshape(-1, 24, 3, 1)
        v3dw = self.v3dws[i].reshape(-1, 3)
        tranc = self.trans[i].reshape(-1, 3)
        # s_tran = self.s_trans[i].reshape(-1, 3)
        # s_tran[1:] = s_tran[0]

        Rcw = Tcw[:, :3, :3]
        Rrw = pose[:, 0].transpose(1, 2).clone()
        orir = Rrw.unsqueeze(1).matmul(oriw).reshape(-1, 6, 3, 3)
        accr = Rrw.unsqueeze(1).matmul(accw)
        oric = Rcw.unsqueeze(1).matmul(oriw).reshape(-1, 6, 3, 3)
        accc = Rcw.unsqueeze(1).matmul(accw)
        j3dr = Rrw.unsqueeze(1).matmul(j3dw).reshape(-1, 24, 3)
        j3dc = Rcw.unsqueeze(1).matmul(j3dw).reshape(-1, 24, 3)
        j3dr = j3dr[:, 1:] - j3dr[:, :1]
        j3dc = j3dc[:, 1:] - j3dc[:, :1]
        v3dr = Rrw.matmul(v3dw.unsqueeze(-1)).squeeze(-1)

        pose[:, 0] = torch.matmul(Rcw, pose[:, 0])
        posec = art.math.rotation_matrix_to_r6d(body_model.forward_kinematics_R(pose)).view(-1, 24, 6).clone()

        pose[:, 0] = torch.eye(3)
        pose = art.math.rotation_matrix_to_r6d(body_model.forward_kinematics_R(pose)).view(-1, 24, 6)
        # todo: use rotation or I
        # orir[:, -1] = oric[:, -1]
        item['oric'] = oric.reshape(-1, 6, 3, 3)
        item['accc'] = accc.reshape(-1, 6, 3)
        item['pose'] = pose.reshape(-1, 24, 6)
        item['orir'] = orir.reshape(-1, 6, 3, 3)
        item['accr'] = accr.reshape(-1, 6, 3)
        item['posec'] = posec.reshape(-1, 24, 6)
        item['j3dr'] = j3dr.reshape(-1, 23, 3)
        item['j3dc'] = j3dc.reshape(-1, 23, 3)
        item['v3dr'] = v3dr.reshape(-1, 3)
        if self.load_run_3d:
            item['j3dr_run'] = self.j3dr_runs[i].reshape(-1, 23, 3)
        item['tranc'] = tranc.reshape(-1, 3)
        # item['s_tranc'] = s_tran.reshape(-1, 3)
        if self.down_sample:
            item = {k: v[::2] for k, v in item.items()}
        if self.names is not None and len(self.names) > 0:
            item['name'] = self.names[i]
        return item

    def __len__(self):
        return len(self.j2dcs)
