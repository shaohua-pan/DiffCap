import torch
import os
import tqdm
from torch.utils.data import Dataset
import articulate as art
import config
from utils.kp_utils import *
from config import device, body_model


class TotalCaptureDataset(Dataset):
    """
        TotalCapture dataset. Adapted for different kinds of data.
    """
    def __init__(self, data_dir, kind, split_size=-1, return_mp=True):
        super(TotalCaptureDataset, self).__init__()
        print('Reading %s dataset "%s"' % (kind, data_dir))
        self.return_mp = return_mp
        dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        j2dc_mps, oriws, accws, poses, trans, Tcws, j2dcs, j2dc_mps2, j3dws, j2dc_mp_origins, trancs, j2dc_mps3 = [], [], [], [], [], [], [], [], [], [], [], []
        self.j2dc_mps, self.oriws, self.accws, self.poses, self.trans, self.Tcws, self.j2dcs, self.j2dc_mps2, self.j3dws, self.j2dc_mp_origins, self.trancs, self.j2dc_mps3 = [], [], [], [], [], [], [], [], [], [], [], []
        for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
            for j in range(8):  # jth camera view
                # i = 0
                if dataset['joint2d_mp'][i][j] is None: continue
                Tcw = dataset['cam_T'][i][j]
                oriw = dataset['imu_ori'][i]
                accw = dataset['imu_acc'][i]
                j3dw = dataset['joint3d'][i]

                Kinv = dataset['cam_K'][i][j].inverse()

                j3dc_mp = Tcw.matmul(art.math.append_one(dataset['joint3d_mp'][i]).unsqueeze(-1)).squeeze(-1)[..., :3]
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
                j2dc_mp = j2dc_mp[:, 1:]

                pose = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i]).view(-1, 24, 3, 3)
                tranc = Tcw.matmul(art.math.append_one(dataset['tran'][i]).unsqueeze(-1)).squeeze(-1)[..., :3]

                j2dc_mps.append(j2dc_mp)
                j2dc_mps2.append(j2dc_mp2)
                j2dc_mps3.append(j2dc_mp3)
                oriws.append(oriw)
                accws.append(accw)
                poses.append(pose)
                Tcws.append(Tcw.repeat(len(pose), 1, 1))
                j2dcs.append(j2dc)
                j3dws.append(j3dw)
                j2dc_mp_origins.append(j2dc_mp_origin)
                trancs.append(tranc)
                trans.append(dataset['tran'][i])

        if split_size > 0:
            for t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12 in zip(j2dc_mps, oriws, accws, poses, Tcws, j2dcs, j2dc_mps2, j3dws, j2dc_mp_origins, trancs, trans, j2dc_mps3):
                if len(t1) < split_size: continue
                m1 = list(t1.split(split_size))
                m1[-1] = t1[-split_size:]
                self.j2dc_mps.extend(m1)
                m2 = list(t2.split(split_size))
                m2[-1] = t2[-split_size:]
                self.oriws.extend(m2)
                m3 = list(t3.split(split_size))
                m3[-1] = t3[-split_size:]
                self.accws.extend(m3)
                m4 = list(t4.split(split_size))
                m4[-1] = t4[-split_size:]
                self.poses.extend(m4)
                m5 = list(t5.split(split_size))
                m5[-1] = t5[-split_size:]
                self.Tcws.extend(m5)
                m6 = list(t6.split(split_size))
                m6[-1] = t6[-split_size:]
                self.j2dcs.extend(m6)
                m7 = list(t7.split(split_size))
                m7[-1] = t7[-split_size:]
                self.j2dc_mps2.extend(m7)
                m8 = list(t8.split(split_size))
                m8[-1] = t8[-split_size:]
                self.j3dws.extend(m8)
                m9 = list(t9.split(split_size))
                m9[-1] = t9[-split_size:]
                self.j2dc_mp_origins.extend(m9)
                m10 = list(t10.split(split_size))
                m10[-1] = t10[-split_size:]
                self.trancs.extend(m10)
                m11 = list(t11.split(split_size))
                m11[-1] = t11[-split_size:]
                self.trans.extend(m11)
                m12 = list(t12.split(split_size))
                m12[-1] = t12[-split_size:]
                self.j2dc_mps3.extend(m12)

        else:
            self.j2dc_mps, self.oriws, self.accws, self.poses, self.Tcws, self.j2dcs, self.j2dc_mps2, self.j3dws, self.j2dc_mp_origins, self.trancs, self.trans, self.j2dc_mps3 = j2dc_mps, oriws, accws, poses, Tcws, j2dcs, j2dc_mps2, j3dws, j2dc_mp_origins, trancs, trans, j2dc_mps3

    def __getitem__(self, i):
        item = {}
        if self.return_mp:
            item['j2dc_mp'] = self.j2dc_mps[i]
            item['j2dc_mp2'] = self.j2dc_mps2[i]
            item['j2dc_mp3'] = self.j2dc_mps3[i]
            item['j2dc_mp_origin'] = self.j2dc_mp_origins[i]
        oriw = self.oriws[i].reshape(-1, 6, 3, 3)
        accw = self.accws[i].reshape(-1, 6, 3, 1)
        pose = self.poses[i]
        Tcw = self.Tcws[i]
        j2dc = self.j2dcs[i]
        j3dw = self.j3dws[i].reshape(-1, 24, 3, 1)
        tranc = self.trancs[i].reshape(-1, 3)
        tran = self.trans[i].reshape(-1, 3)

        Rcw = Tcw[:, :3, :3]
        Rrw = pose[:, 0].transpose(1, 2)
        orir = Rrw.unsqueeze(1).matmul(oriw)
        accr = Rrw.unsqueeze(1).matmul(accw)
        oric = Rcw.unsqueeze(1).matmul(oriw)
        accc = Rcw.unsqueeze(1).matmul(accw)
        j3dr = Rrw.unsqueeze(1).matmul(j3dw).reshape(-1, 24, 3)
        j3dr = j3dr[:, 1:] - j3dr[:, :1]

        pose[:, 0] = torch.matmul(Rcw, pose[:, 0])
        posec = art.math.rotation_matrix_to_r6d(body_model.forward_kinematics_R(pose)).view(-1, 24, 6).clone()

        pose[:, 0] = torch.eye(3)
        pose = art.math.rotation_matrix_to_r6d(body_model.forward_kinematics_R(pose)).view(-1, 24, 6)
        # orir[:, -1] = oric[:, -1]
        item['oric'] = oric.reshape(-1, 6, 3, 3)
        item['accc'] = accc.reshape(-1, 6, 3)
        item['pose'] = pose.reshape(-1, 24, 6)
        item['orir'] = orir.reshape(-1, 6, 3, 3)
        item['accr'] = accr.reshape(-1, 6, 3)
        item['posec'] = posec.reshape(-1, 24, 6)
        item['j2dc'] = j2dc.reshape(-1, 33, 2)
        item['j3dr'] = j3dr.reshape(-1, 23, 3)
        item['tranc'] = tranc.reshape(-1, 3)
        item['tran'] = tran.reshape(-1, 3)
        return item

    def __len__(self):
        return len(self.oriws)
