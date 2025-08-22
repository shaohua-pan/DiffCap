# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
if __name__ == '__main__':
    import sys
    os.chdir('..')
    sys.path.insert(0, os.getcwd())
import json
import config
from utils.parser_util import train_args
from dataset.aist_dataset import AISTDataset
from dataset.amass_dataset import AMASSDataset
from torch.utils.data import DataLoader, ConcatDataset
from train.training_loop import TrainLoop
from utils.model_util import create_model_and_diffusion, fixseed, collate_fn
from config import device
import torch
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation


def train_2d_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, oric, accc = item['j2dc'], item['oric'], item['accc']
    j2dc = j2dc.reshape(bs, j2dc.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = j2dc.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['imu'] = torch.cat((oric.flatten(2), accc.flatten(2)), dim=2).reshape(bs, -1)
    # todo: whether use noise?
    # condition['y']['imu'] = torch.normal(condition['y']['imu'], 0.01)
    return motion, condition

def train_pose_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, oric, accc, pose = item['j2dc'], item['oric'], item['accc'], item['posec']
    pose = pose.reshape(bs, pose.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = pose.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['2d'] = j2dc.reshape(bs, -1)
    # condition['y']['2d'] = torch.normal(condition['y']['2d'], 0.01)
    return motion, condition

def train_pose_no_cond_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, oric, accc, pose = item['j2dc'], item['oric'], item['accc'], item['posec']
    pose = pose.reshape(bs, pose.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = pose.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    return motion, condition

def train_pose_acc_cond_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, orir, accr, pose = item['j2dc'], item['orir'], item['accr'], item['pose']
    pose = pose.reshape(bs, pose.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = pose.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['acc'] = accr.reshape(bs, -1)
    return motion, condition

def train_pose_2d_acc_cond_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, oric, accc, pose = item['j2dc'], item['orir'], item['accr'], item['pose']
    pose = pose.reshape(bs, pose.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = pose.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['2d_acc'] = torch.cat((j2dc.flatten(2), accc.flatten(2)), dim=2).reshape(bs, -1)
    return motion, condition

def train_2d_slide_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, oric, accc = item['j2dc'], item['oric'], item['accc']
    j2dc = j2dc.reshape(bs, j2dc.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = j2dc.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['imu'] = torch.cat((oric.flatten(2), accc.flatten(2)), dim=2).reshape(bs, -1)
    condition['sliding'] = True
    # condition['y']['imu'] = torch.normal(condition['y']['imu'], 0.01)
    return motion, condition

def train_dual_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, oric, accc, pose = item['j2dc'], item['oric'], item['accc'], item['posec']
    pose = pose.reshape(bs, pose.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    j2dc = j2dc.reshape(bs, j2dc.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = (j2dc.permute(0, 2, 3, 1), pose.permute(0, 2, 3, 1))  # (bs, njoints, nfeats, nframes)
    condition = {}
    return motion, condition

def train_2d_mp_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, oric, accc, j2dc_mp = item['j2dc'], item['oric'], item['accc'], item['j2dc_mp']
    j2dc = j2dc.reshape(bs, j2dc.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = j2dc.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    return motion, condition

def train_2d_mp_imu_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, oric, accc, j2dc_mp = item['j2dc'], item['oric'], item['accc'], item['j2dc_mp']
    j2dc = j2dc.reshape(bs, j2dc.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = j2dc.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['mp_imu'] = torch.cat((j2dc_mp.flatten(2), oric.flatten(2), accc.flatten(2)), dim=2).reshape(bs, -1)
    return motion, condition

def train_2d_mp_imu_concat_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, oric, accc, j2dc_mp = item['j2dc'], item['oric'], item['accc'], item['j2dc_mp']
    j2dc = j2dc.reshape(bs, j2dc.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = j2dc.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['mp_imu'] = torch.cat((j2dc_mp.flatten(2), oric.flatten(2), accc.flatten(2)), dim=2).reshape(bs, -1)
    return motion, condition

def train_diffusion_2d_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, oric, accc, j2dc_mp = item['j2dc'], item['oric'], item['accc'], item['j2dc_mp']
    j2dc = j2dc.reshape(bs, j2dc.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = j2dc.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    condition['y']['imu'] = torch.cat((oric.flatten(2), accc.flatten(2)), dim=2).reshape(bs, -1)
    return motion, condition

def train_diffusion_3d_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    # TODO: comment this after testing
    # j2dc, orir, accr, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['j2dc_mp'], item['j3dr']
    j2dc, orir, accr, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['j2dc_mp2'], item['j3dr']
    j3dr = j3dr.reshape(bs, j2dc.shape[1], 23, 3)  # (bs, nframes, njoints, nfeats)
    motion = j3dr.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    condition['y']['imu'] = torch.cat((orir.flatten(2), accr.flatten(2)), dim=2).reshape(bs, -1)
    return motion, condition

def train_diffusion_3d_wo_diff_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, orir, accr, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['j2dc_mp2'], item['j3dr']
    j3dr = j3dr.reshape(bs, j2dc.shape[1], 23, 3)  # (bs, nframes, njoints, nfeats)
    motion = j3dr.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    condition['y']['imu'] = torch.cat((orir.flatten(2), accr.flatten(2)), dim=2).reshape(bs, -1)
    return motion, condition

def train_diffusion_3d_exchange_input_cond_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    # TODO: comment this after testing
    # j2dc, orir, accr, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['j2dc_mp'], item['j3dr']
    j2dc, orir, accr, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['j2dc_mp2'], item['j3dr']
    j3dr = j3dr.reshape(bs, j2dc.shape[1], 23, 3)  # (bs, nframes, njoints, nfeats)
    motion = j3dr.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    condition['y']['imu'] = torch.cat((orir.flatten(2), accr.flatten(2)), dim=2).reshape(bs, -1)
    return motion, condition

def train_joint_diffusion_model_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    oric, accc, j2dc_mp, j3dc = item['oric'], item['accc'], item['j2dc_mp2'], item['j3dc']
    j3dc = j3dc.reshape(bs, j2dc_mp.shape[1], 23, 3)  # (bs, nframes, njoints, nfeats)
    motion = j3dc.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    condition['y']['imu'] = torch.cat((oric.flatten(2), accc.flatten(2)), dim=2).reshape(bs, -1)
    return motion, condition

def train_diffusion_3d_wo_input_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, orir, accr, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['j2dc_mp2'], item['j3dr']
    j3dr = j3dr.reshape(bs, j2dc.shape[1], 23, 3)  # (bs, nframes, njoints, nfeats)
    motion = j3dr.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['mp_imu'] = torch.cat((j2dc_mp.flatten(2), orir.flatten(2), accr.flatten(2)), dim=2).reshape(bs, -1)
    return motion, condition


def train_diffusion_3d_wo_cond_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, orir, accr, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['j2dc_mp2'], item['j3dr']
    j3dr = j3dr.reshape(bs, j2dc.shape[1], 23, 3)  # (bs, nframes, njoints, nfeats)
    motion = j3dr.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    condition['y']['imu'] = torch.cat((orir.flatten(2), accr.flatten(2)), dim=2).reshape(bs, -1)
    return motion, condition


def train_diffusion_pose_1step_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    # j2dc, orir, accr, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['j2dc_mp'], item['j3dr']
    j2dc, orir, accr, j2dc_mp, pose = item['j2dc'], item['orir'], item['accr'], item['j2dc_mp2'], item['pose']
    pose = pose.reshape(bs, j2dc.shape[1], 24, 6)  # (bs, nframes, njoints, nfeats)
    motion = pose.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    condition['y']['imu'] = torch.cat((orir.flatten(2), accr.flatten(2)), dim=2).reshape(bs, -1)
    return motion, condition


def train_diffusion_3d_slide_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    # TODO: comment this after testing
    # j2dc, orir, accr, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['j2dc_mp'], item['j3dr']
    j2dc, orir, accr, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['j2dc_mp2'], item['j3dr']
    j3dr = j3dr.reshape(bs, j2dc.shape[1], 23, 3)  # (bs, nframes, njoints, nfeats)
    motion = j3dr.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    condition['y']['imu'] = torch.cat((orir.flatten(2), accr.flatten(2)), dim=2).reshape(bs, -1)
    condition['sliding'] = True
    return motion, condition

def train_diffusion_2d_imur_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, oric, accc, j2dc_mp = item['j2dc'], item['oric'], item['accc'], item['j2dc_mp']
    root_imu = oric[:, :, -1:, :]
    oric[:, :, :-1, :] = torch.matmul(root_imu.transpose(-1, -2), oric[:, :, :-1, :])
    accc[:, :, :, :] = torch.matmul(root_imu.transpose(-1, -2), accc[:, :, :, :].unsqueeze(-1)).squeeze(-1)
    j2dc = j2dc.reshape(bs, j2dc.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = j2dc.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    condition['y']['imu'] = torch.cat((oric.flatten(2), accc.flatten(2)), dim=2).reshape(bs, -1)
    return motion, condition

def train_diffusion_pose_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, orir, accr, pose, j2dc_mp = item['j2dc'], item['orir'], item['accr'], item['pose'], item['j2dc_mp']
    pose = pose.reshape(bs, pose.shape[1], 24, 6)  # (bs, nframes, njoints, nfeats)
    motion = pose.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['2d_imu'] = torch.cat((j2dc.flatten(2), orir.flatten(2), accr.flatten(2)), dim=2).reshape(bs, -1)
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    return motion, condition

def train_diffusion_pose_3dcond_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    if 'j3dr_run' not in item:
        j2dc, orir, accr, pose, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['pose'], item['j2dc_mp'], item['j3dr']
    else:
        j2dc, orir, accr, pose, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['pose'], item['j2dc_mp'], item['j3dr_run']
    pose = pose.reshape(bs, pose.shape[1], 24, 6)  # (bs, nframes, njoints, nfeats)
    motion = pose.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['3d_imu'] = torch.cat((j3dr.flatten(2), orir.flatten(2), accr.flatten(2)), dim=2).reshape(bs, -1)
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    return motion, condition


def train_diffusion_pose_attn_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    if 'j3dr_run' not in item:
        j2dc, orir, accr, pose, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['pose'], item['j2dc_mp'], item['j3dr']
    else:
        j2dc, orir, accr, pose, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['pose'], item['j2dc_mp'], item['j3dr_run']
    pose = pose.reshape(bs, pose.shape[1], 24, 6)  # (bs, nframes, njoints, nfeats)
    motion = pose.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['3d_imu'] = torch.cat((j3dr.flatten(2), orir.flatten(2), accr.flatten(2)), dim=2).reshape(bs, -1)
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    return motion, condition

def train_diffusion_pose_edit_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, accr, pose = item['j2dc'], item['accr'], item['pose']
    pose = pose.reshape(bs, pose.shape[1], 24 * 6)  # (bs, nframes, njoints * nfeats)
    j2dc = j2dc.reshape(bs, j2dc.shape[1], 33 * 2)
    accr = accr.reshape(bs, accr.shape[1], 6 * 3)
    motion = torch.cat((pose, j2dc, accr), dim=2).reshape(bs, pose.shape[1], 1, -1).permute(0, 2, 3, 1) # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    return motion, condition

def train_diffusion_pose_3dcond_wo_diff_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    # j2dc, orir, accr, pose, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['pose'], item['j2dc_mp'], item['j3dr']
    j2dc, orir, accr, pose, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['pose'], item['j2dc_mp'], item['j3dr_run']
    pose = pose.reshape(bs, pose.shape[1], 24, 6)  # (bs, nframes, njoints, nfeats)
    motion = pose.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['3d_imu'] = torch.cat((j3dr.flatten(2), orir.flatten(2), accr.flatten(2)), dim=2).reshape(bs, -1)
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    return motion, condition

def train_diffusion_tran_3dcond_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    # j2dc, orir, accr, pose, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['pose'], item['j2dc_mp'], item['j3dr']
    j2dc, orir, accr, v3dr, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['v3dr'], item['j2dc_mp'], item['j3dr_run']
    v3dr = v3dr.reshape(bs, v3dr.shape[1], 1, 3)  # (bs, nframes, njoints, nfeats)
    motion = v3dr.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['3d_imu'] = torch.cat((j3dr.flatten(2), orir.flatten(2), accr.flatten(2)), dim=2).reshape(bs, -1)
    # condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    return motion, condition

def train_diffusion_tran_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    # j2dc, orir, accr, pose, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['pose'], item['j2dc_mp'], item['j3dr']
    j2dc, orir, accr, v3dr, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['v3dr'], item['j2dc_mp3'], item['j3dr_run']
    v3dr = v3dr.reshape(bs, v3dr.shape[1], 1, 3)  # (bs, nframes, njoints, nfeats)
    motion = v3dr.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['3d_imu'] = torch.cat((j3dr.flatten(2), orir.flatten(2), accr.flatten(2)), dim=2).reshape(bs, -1)
    condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    return motion, condition

def train_diffusion_tran_pos_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    # j2dc, orir, accr, pose, j2dc_mp, j3dr = item['j2dc'], item['orir'], item['accr'], item['pose'], item['j2dc_mp'], item['j3dr']
    j2dc, oric, accc, tranc, j2dc_mp, j3dr = item['j2dc'], item['oric'], item['accc'], item['tranc'], item['j2dc_mp3'], item['j3dr_run']
    tranc = tranc.reshape(bs, tranc.shape[1], 1, 3)  # (bs, nframes, njoints, nfeats)
    motion = tranc.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['imu_2d'] = torch.cat((oric.flatten(2), accc.flatten(2), j2dc_mp.flatten(2)), dim=2).reshape(bs, -1)
    return motion, condition

def train_2d_nocond_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, oric, accc, j2dc_mp = item['j2dc'], item['oric'], item['accc'], item['j2dc_mp']
    j2dc = j2dc.reshape(bs, j2dc.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = j2dc.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    # condition['y']['mp'] = j2dc_mp.reshape(bs, -1)
    return motion, condition

def train_pose_acc_cond_concat_parser(item, bs):
    item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}
    j2dc, orir, accr, pose = item['j2dc'], item['orir'], item['accr'], item['pose']
    pose = pose.reshape(bs, pose.shape[1], -1, 2)  # (bs, nframes, njoints, nfeats)
    motion = pose.permute(0, 2, 3, 1)  # (bs, njoints, nfeats, nframes)
    condition = {}
    condition['y'] = {}
    condition['y']['acc'] = accr.reshape(bs, -1)
    return motion, condition

def main():
    # load args
    args = train_args()
    train_type = args.train_type
    print(f'-----------------------Train {train_type}-----------------------')
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    # dump args
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    # load data
    # TODO:
    aist_train_dataset = AISTDataset(config.paths.aist_dir, 'train', args.num_frames, load_run_3d=args.load_run_3d, down_sample=args.down_sample)
    amass_train_dataset = AMASSDataset(config.paths.amass_dir, 'train', args.num_frames, use_random_cam=False, load_run_3d=args.load_run_3d, downsample=args.down_sample)
    aist_val_dataset = AISTDataset(config.paths.aist_dir, 'val', args.num_frames, load_run_3d=args.load_run_3d, down_sample=args.down_sample)
    amass_val_dataset = AMASSDataset(config.paths.amass_dir, 'val', args.num_frames, use_random_cam=False, load_run_3d=args.load_run_3d, downsample=args.down_sample)
    # amass_test_dataset = AMASSDataset(config.paths.amass_dir, 'test', args.num_frames, use_random_cam=False, load_run_3d=args.load_run_3d, downsample=args.down_sample)
    num_workers = 8
    if sys.platform.startswith('win'):
        num_workers = 0
    print(f'num_workers: {num_workers}')
    # todo: use all dataset as training data and no valid ?
    train_dataloader = DataLoader(ConcatDataset([aist_train_dataset, amass_train_dataset, aist_val_dataset, amass_val_dataset]), batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    # train_dataloader = DataLoader(amass_test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    data_parser = eval('train_{}_parser'.format(train_type))
    # create model and diffusion
    model, diffusion = create_model_and_diffusion(args, train_type)
    model.to(device)

    # train
    TrainLoop(args, train_platform, model, diffusion, data_parser, train_dataloader).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
