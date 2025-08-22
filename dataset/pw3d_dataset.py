import torch
import os
import tqdm
from torch.utils.data import Dataset
import articulate as art
import config
from utils.kp_utils import *
from config import device, body_model


class Pw3dDataset(Dataset):
    """
        3DPW dataset. Adapted for different kinds of data.
    """
    def __init__(self, data_dir, kind, split_size=-1, return_mp=True):
        super(Pw3dDataset, self).__init__()
        print('Reading %s dataset "%s"' % (kind, data_dir))
        self.return_mp = return_mp
        dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        j2dc_mps, posecs, Tcws, j2dc_mps2, j2dc_mp_origins, trancs, orics, acccs, j3dcs = [], [], [], [], [], [], [], [], []
        self.j2dc_mps, self.posecs, self.Tcws, self.j2dc_mps2, self.j2dc_mp_origins, self.trancs, self.orics, self.acccs, self.j3dcs = [], [], [], [], [], [], [], [], []
        for i in tqdm.trange(len(dataset['posec'])):  # ith sequence
            Tcw = dataset['cam_T'][i]
            oric = dataset['imu_oric'][i]
            accc = dataset['imu_accc'][i]

            Kinv = dataset['cam_K'][i].inverse()

            j2dc_mp = torch.zeros(len(oric), 33, 3)
            j2dc_mp[..., :2] = dataset['joint2d_mp'][i][..., :2]
            j2dc_mp[..., 0] = j2dc_mp[..., 0] * 1920
            j2dc_mp[..., 1] = j2dc_mp[..., 1] * 1080
            j2dc_mp_origin = j2dc_mp.clone()
            j2dc_mp_origin[..., -1] = dataset['joint2d_mp'][i][..., -1]

            j2dc_mp = Kinv.matmul(art.math.append_one(j2dc_mp[..., :2]).unsqueeze(-1)).squeeze(-1)
            j2dc_mp[..., :2] = j2dc_mp[..., :2] / (get_bbox_scale(j2dc_mp)).view(-1, 1, 1)
            j2dc_mp2 = j2dc_mp.clone()
            j2dc_mp[:, 1:, :2] = j2dc_mp[:, 1:, :2] - j2dc_mp[:, :1, :2]
            j2dc_mp2[:, 24:, :2] = j2dc_mp2[:, 24:, :2] - j2dc_mp2[:, 23:24, :2]
            j2dc_mp2[:, :23, :2] = j2dc_mp2[:, :23, :2] - j2dc_mp2[:, 23:24, :2]
            j2dc_mp[..., -1] = dataset['joint2d_mp'][i][..., -1]
            j2dc_mp2[..., -1] = dataset['joint2d_mp'][i][..., -1]
            j2dc_mp = j2dc_mp[:, 1:]

            posec = dataset['posec'][i].view(-1, 24, 3, 3)
            tranc = dataset['tranc'][i].view(-1, 3)
            j3dc = dataset['joint3d'][i]

            j2dc_mps.append(j2dc_mp)
            j2dc_mps2.append(j2dc_mp2)
            orics.append(oric)
            acccs.append(accc)
            posecs.append(posec)
            j2dc_mp_origins.append(j2dc_mp_origin)
            trancs.append(tranc)
            j3dcs.append(j3dc)

        self.j2dc_mps, self.posecs, self.Tcws, self.j2dc_mps2, self.j2dc_mp_origins, self.trancs, self.orics, self.acccs, self.j3dcs =\
            j2dc_mps, posecs, Tcws, j2dc_mps2, j2dc_mp_origins, trancs, orics, acccs, j3dcs

    def __getitem__(self, i):
        item = {}
        if self.return_mp:
            item['j2dc_mp'] = self.j2dc_mps[i]
            item['j2dc_mp2'] = self.j2dc_mps2[i]
            item['j2dc_mp_origin'] = self.j2dc_mp_origins[i]
        oric = self.orics[i].reshape(-1, 6, 3, 3)
        accc = self.acccs[i].reshape(-1, 6, 3, 1)
        posec = self.posecs[i]
        tranc = self.trancs[i].reshape(-1, 3)
        j3dc = self.j3dcs[i].reshape(-1, 24, 3, 1)

        Rrc = oric[:, -1].transpose(1, 2)
        orir = Rrc.unsqueeze(1).matmul(oric)
        accr = Rrc.unsqueeze(1).matmul(accc)
        pose = posec.clone()
        pose[:, 0] = torch.eye(3)
        poser = art.math.rotation_matrix_to_r6d(body_model.forward_kinematics_R(pose)).view(-1, 24, 6)
        posec = art.math.rotation_matrix_to_r6d(body_model.forward_kinematics_R(posec)).view(-1, 24, 6)

        j3dr = Rrc.unsqueeze(1).matmul(j3dc).reshape(-1, 24, 3)
        j3dr = j3dr[:, 1:] - j3dr[:, :1]
        item['oric'] = oric.reshape(-1, 6, 3, 3)
        item['accc'] = accc.reshape(-1, 6, 3)
        item['posec'] = posec.reshape(-1, 24, 6)
        item['tranc'] = tranc.reshape(-1, 3)
        item['orir'] = orir.reshape(-1, 6, 3, 3)
        item['accr'] = accr.reshape(-1, 6, 3)
        item['j3dr'] = j3dr.reshape(-1, 23, 3)
        item['pose'] = poser.reshape(-1, 24, 6)
        return item

    def __len__(self):
        return len(self.orics)
