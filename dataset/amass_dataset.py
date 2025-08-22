import torch
import os
import tqdm
from torch.utils.data import Dataset
import articulate as art
import random
from config import device, body_model
from utils.kp_utils import *


class AMASSDataset(Dataset):
    """
        AMASS dataset. Adapted for different kinds of data.
    """
    def __init__(self, data_dir, kind, split_size=-1, return_mp=True, use_random_cam=True, load_run_3d=False, downsample=False):
        super(AMASSDataset, self).__init__()
        print('Reading %s dataset "%s"' % (kind, data_dir))
        dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        self.downsample = downsample
        if downsample:
            split_size *= 2
        self.cam = []
        if not use_random_cam:
            aist_dataset = torch.load(os.path.join(config.paths.aist_dir, kind + '.pt'))
            tc_dataset = torch.load(os.path.join(config.paths.totalcapture_dir, 'test.pt'))
            for j in range(9):
                self.cam.append(aist_dataset['cam_T'][0][j][:3, :3].clone())
            for j in range(8):
                self.cam.append(tc_dataset['cam_T'][0][j][:3, :3].clone())
        accws, oriws, j3dw_mps, poses, j3dws, names, j3dr_runs, v3dws, s_tranws = [], [], [], [], [], [], [], [], []
        self.return_mp = return_mp
        self.load_run_3d = load_run_3d
        self.conf = torch.load('data/dataset_work/syn_c.pt')
        self.accws, self.oriws, self.j3dw_mps, self.poses, self.j3dws, self.names, self.j3dr_runs, self.v3dws, self.s_tranws = [], [], [], [], [], [], [], [], []
        dataset2 = None
        if load_run_3d:
            assert kind != 'test'
            if kind == 'val':
                dataset2 = torch.load(os.path.join(data_dir, 'val_3d.pt'))
            else:
                d3_1 = torch.load(os.path.join(data_dir, 'train_3d_0-3000.pt'))
                d3_2 = torch.load(os.path.join(data_dir, 'train_3d_3000-6000.pt'))
                d3_3 = torch.load(os.path.join(data_dir, 'train_3d_6000-9000.pt'))
                dataset2 = {**d3_1, **d3_2, **d3_3}
        for i in tqdm.trange(len(dataset['imu_acc'])):
            accw = dataset['imu_acc'][i]  # N, 6, 3
            oriw = dataset['imu_ori'][i]  # N, 6, 3, 3
            root = dataset['joint3d'][i][0, 0].clone()
            j3dw_mp = dataset['sync_3d_mp'][i] - root  # N, 33, 3
            j3dw = dataset['joint3d'][i] - root
            s_tranw = j3dw[:, 0].clone()
            s_tranw[1:] = j3dw[:, 0][:-1]
            v3dw = (dataset['joint3d'][i][2:] - dataset['joint3d'][i][:-2]) * 30
            v3dw = torch.cat((torch.zeros(1, 3), v3dw[:, 0], torch.zeros(1, 3)), dim=0) / config.vel_scale
            pose = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i]).view(-1, 24, 3, 3)
            name = dataset['name'][i]
            # j3dr_run = [None for _ in range(pose.shape[0])]
            j3dr_run = j3dw
            if load_run_3d and name in dataset2:
                j3dr_run = dataset2[name]

            accws.append(accw)
            oriws.append(oriw)
            j3dw_mps.append(j3dw_mp)
            j3dws.append(j3dw)
            s_tranws.append(s_tranw)
            poses.append(pose)
            names.append(name)
            j3dr_runs.append(j3dr_run)
            v3dws.append(v3dw)
            # break

        if split_size > 0:
            for t1, t2, t3, t4, t5, t6, t7, t8 in zip(accws, oriws, j3dw_mps, poses, j3dws, j3dr_runs, v3dws, s_tranws):
                # if len(t1) < split_size or t6[0] is None: continue
                if len(t1) < split_size: continue
                m1 = list(t1.split(split_size))
                m1[-1] = t1[-split_size:]
                self.accws.extend(m1)
                m2 = list(t2.split(split_size))
                m2[-1] = t2[-split_size:]
                self.oriws.extend(m2)
                m3 = list(t3.split(split_size))
                m3[-1] = t3[-split_size:]
                self.j3dw_mps.extend(m3)
                m4 = list(t4.split(split_size))
                m4[-1] = t4[-split_size:]
                self.poses.extend(m4)
                m5 = list(t5.split(split_size))
                m5[-1] = t5[-split_size:]
                self.j3dws.extend(m5)
                m6 = list(t6.split(split_size))
                m6[-1] = t6[-split_size:]
                self.j3dr_runs.extend(m6)
                m7 = list(t7.split(split_size))
                m7[-1] = t7[-split_size:]
                self.v3dws.extend(m7)
                m8 = list(t8.split(split_size))
                m8[-1] = t8[-split_size:]
                self.s_tranws.extend(m8)
        else:
            self.accws, self.oriws, self.j3dw_mps, self.poses, self.j3dws, self.j3dr_runs, self.v3dws, self.s_tranws = accws, oriws, j3dw_mps, poses, j3dws, j3dr_runs, v3dws, s_tranws
        if split_size < 0:
            self.names = names
        assert len(self.accws) == len(self.oriws) == len(self.j3dw_mps) == len(self.poses) == len(self.j3dws) == len(self.j3dr_runs) == len(self.v3dws)

    def __getitem__(self, i):
        item = {}
        accw = self.accws[i].reshape(-1, 6, 3, 1)
        oriw = self.oriws[i].reshape(-1, 6, 3, 3)
        j3dw_mp = self.j3dw_mps[i].reshape(-1, 33, 3, 1)
        j3dw = self.j3dws[i].reshape(-1, 24, 3, 1)
        pose = self.poses[i].reshape(-1, 24, 3, 3)
        v3dw = self.v3dws[i].reshape(-1, 3)
        s_tranw = self.s_tranws[i].reshape(-1, 3)
        Rrw = pose[:, 0].transpose(1, 2).clone()

        # Generate random rotation matrix as the camera rotation
        if len(self.cam) > 0:
            Rcw = self.cam[random.randint(0, len(self.cam) - 1)]
        else:
            Rwc0 = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1.]])
            Rc0c = art.math.generate_random_rotation_matrix_constrained(n=1, y=(-180, 180), p=(-30, 30), r=(-5, 5))[0]
            Rcw = Rwc0.mm(Rc0c).t()

        accc = Rcw.matmul(accw)
        oric = Rcw.matmul(oriw).reshape(-1, 6, 3, 3)
        accr = Rrw.unsqueeze(1).matmul(accw)
        orir = Rrw.unsqueeze(1).matmul(oriw).reshape(-1, 6, 3, 3)
        j3dc_mp = Rcw.matmul(j3dw_mp).squeeze(-1)
        j3dc = Rcw.matmul(j3dw).squeeze(-1)

        pose[:, 0] = Rcw.matmul(pose[:, 0])
        posec = art.math.rotation_matrix_to_r6d(body_model.forward_kinematics_R(pose)).view(-1, 24, 6).clone()

        pose[:, 0] = torch.eye(3)
        pose = art.math.rotation_matrix_to_r6d(body_model.forward_kinematics_R(pose)).view(-1, 24, 6)

        # Randomly translate the motion
        random_tranc = art.math.lerp(torch.tensor([-1, -1, 3.]), torch.tensor([1, 1, 8.]), torch.rand(3))
        random_tranc[2] -= j3dc[..., -1].min()
        # s_tranc = Rcw.matmul(s_tranw.unsqueeze(-1)).squeeze(-1)
        # s_tranc = s_tranc + random_tranc
        # s_tranc[1:] = s_tranc[0]
        j3dc = j3dc + random_tranc
        tranc = j3dc[:, 0].clone()
        j3dw = Rcw.transpose(-1, -2).matmul(j3dc.unsqueeze(-1))
        j3dr = Rrw.unsqueeze(1).matmul(j3dw).squeeze(-1)
        j3dr = j3dr[:, 1:] - j3dr[:, :1]
        v3dr = Rrw.matmul(v3dw.unsqueeze(-1)).squeeze(-1)
        j3dc_mp = j3dc_mp + random_tranc
        j2dc = j3dc_mp / j3dc_mp[..., -1:]
        j2dc_mp = j2dc.clone()
        j2dc = j2dc[..., :2]
        j2dc = j2dc / (get_bbox_scale(j2dc)).view(-1, 1, 1)
        # j2dc = j2dc[:, 1:] - j2dc[:, :1]
        j2dc[:, 24:] = j2dc[:, 24:] - j2dc[:, 23:24]
        j2dc[:, :23] = j2dc[:, :23] - j2dc[:, 23:24]

        ran = range(0, len(self.conf))
        rand = random.sample(ran, len(accc))
        p = self.conf[rand]
        j2dc_mp[..., :2] = torch.normal(j2dc_mp[..., :2], 0.003 * (1 - p))
        j2dc_mp[..., -1:] = p
        j2dc_mp3 = j2dc_mp.clone()
        j2dc_mp[..., :2] = j2dc_mp[..., :2] / (get_bbox_scale(j2dc_mp)).view(-1, 1, 1)
        j2dc_mp2 = j2dc_mp.clone()
        j2dc_mp2[:, 24:, :2] = j2dc_mp2[:, 24:, :2] - j2dc_mp2[:, 23:24, :2]
        j2dc_mp2[:, :23, :2] = j2dc_mp2[:, :23, :2] - j2dc_mp2[:, 23:24, :2]
        j2dc_mp[:, 1:, :2] = j2dc_mp[:, 1:, :2] - j2dc_mp[:, :1, :2]
        j2dc_mp = j2dc_mp[:, 1:]

        j3dc = j3dc[:, 1:] - j3dc[:, :1]

        # todo: use rotation or I
        # orir[:, -1] = oric[:, -1]
        item['accc'] = accc.reshape(-1, 6, 3)
        item['oric'] = oric.reshape(-1, 6, 3, 3)
        item['j2dc'] = j2dc.reshape(-1, 33, 2)
        if self.return_mp:
            item['j2dc_mp'] = j2dc_mp.reshape(-1, 32, 3)
            item['j2dc_mp2'] = j2dc_mp2.reshape(-1, 33, 3)
            item['j2dc_mp3'] = j2dc_mp3.reshape(-1, 33, 3)
        item['pose'] = pose.reshape(-1, 24, 6)
        item['accr'] = accr.reshape(-1, 6, 3)
        item['orir'] = orir.reshape(-1, 6, 3, 3)
        item['posec'] = posec.reshape(-1, 24, 6)
        item['j3dr'] = j3dr.reshape(-1, 23, 3)
        item['j3dc'] = j3dc.reshape(-1, 23, 3)
        if len(self.names) > 0:
            item['name'] = self.names[i]
        if self.load_run_3d:
            item['j3dr_run'] = self.j3dr_runs[i].reshape(-1, 23, 3)
        item['v3dr'] = v3dr.reshape(-1, 3)
        item['tranc'] = tranc.reshape(-1, 3)
        # item['s_tranc'] = s_tranc.reshape(-1, 3)
        if self.downsample:
            item = {k: v[::2] for k, v in item.items()}
        return item

    def __len__(self):
        return len(self.accws)
