import time
import os
if __name__ == '__main__':
    import sys
    os.chdir('..')
    sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from utils.parser_util import edit_args
from utils.model_util import create_model_and_diffusion, fixseed
from model.cfg_sampler import ClassifierFreeSampleModel
from model.contact_rnn import ContactRNN
from model.position_rnn import PositionRNN
from dataset.aist_dataset import AISTDataset
from dataset.total_capture_dataset import TotalCaptureDataset
from dataset.pw3d_dataset import Pw3dDataset
import config
from torch.utils.data import DataLoader
import tqdm
from utils.kp_utils import *
import cv2
import articulate as art
from collections import deque
from model.smplify.run import smplify_runner

device = config.device


def _load_model(train_type='2d', args=None):
    if args is None:
        args = edit_args()
    fixseed(args.seed)
    model, diffusion = create_model_and_diffusion(args, train_type)
    if train_type == 'diffusion_3d' or train_type == 'diffusion_pose_1step' or train_type == 'diffusion_3d_wo_cond' or train_type == 'diffusion_3d_wo_input' or train_type == 'diffusion_3d_exchange_input_cond' or train_type == 'diffusion_3d_wo_diff':
        state_dict = torch.load(args.model_path, map_location='cpu')
    elif train_type == 'diffusion_pose_3dcond' or train_type == 'diffusion_pose_3dcond_wo_diff':
        state_dict = torch.load(args.model_path2, map_location='cpu')
    elif train_type == 'diffusion_tran_3dcond':
        state_dict = torch.load(args.model_path3, map_location='cpu')
    else:
        raise NotImplementedError
    model.load_state_dict(state_dict, strict=False)
    model = ClassifierFreeSampleModel(model)
    model.to(device)
    model.eval()
    return model, diffusion, args


def _vis_overlay(video_path, pose, tran, K, name):
    video = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(os.path.join('data/temp/temp_video', name + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 60, (1920, 1080))
    render = art.Renderer(resolution=(1920, 1080), official_model_file=config.paths.smpl_file)
    f = 0
    while True:
        im = video.read()[1]
        if im is None:
            break
        if f >= len(pose):
            break
        verts = config.body_model_cpu.forward_kinematics(pose[f].view(-1, 24, 3, 3), tran=tran[f].view(-1, 3), calc_mesh=True)[2][0]
        img = render.render(im, verts, K, mesh_color=(.7, .7, .6, 1.))
        cv2.imshow('f', img)
        cv2.waitKey(1)
        writer.write(img)
        f += 1
    writer.release()


def _eval_ds(dataset, dataloader, model_3d, model_tran, model_pose, diffusion_3d, diffusion_pose, args, cam_num=9, debug_name=None, debug_path=None):
    sample_3d_fn = diffusion_3d.ddim_sample_loop
    sample_pt_fn = diffusion_pose.ddim_sample_loop
    pose_p, pose_t, idx, tran_t, tran_p = [], [], -1, [], []

    for item in tqdm.tqdm(dataloader):
        idx += 1
        # if idx != 35*8:
        #     continue
        orirs = item['orir'].squeeze(0).to(device)
        orics = item['oric'].squeeze(0).to(device)
        accrs = item['accr'].squeeze(0).to(device)
        j2ds = item['j2dc_mp2'].squeeze(0).to(device)
        posec = config.body_model_cpu.inverse_kinematics_R(art.math.r6d_to_rotation_matrix(item['posec']).reshape(-1, 24, 3, 3))
        tranc = item['tranc'].squeeze(0)

        cam_k = dataset['cam_K'][idx // cam_num][idx % cam_num]

        pose_t.append(posec)
        tran_t.append(tranc)

        cur_frame = 0
        sample_3ds, temp, sample_poses, sample_trans = [], [], [], []

        # sample 3d
        while cur_frame < len(orics):
            if cur_frame == 0:
                cur_frame += 60
            else:
                cur_frame += 30
            condition = {}
            condition['y'] = {}
            condition['y']['mp_imu'] = torch.cat((j2ds[cur_frame - 60:cur_frame].flatten(1), orirs[cur_frame - 60:cur_frame].flatten(1), accrs[cur_frame - 60:cur_frame].flatten(1)), dim=1).reshape(1, -1)
            condition['y']['mp'] = j2ds[cur_frame - 60:cur_frame].reshape(1, -1)
            condition['y']['imu'] = torch.cat((orirs[cur_frame - 60:cur_frame].flatten(1), accrs[cur_frame - 60:cur_frame].flatten(1)), dim=1).reshape(1, -1)
            condition['y']['scale'] = torch.ones(1, device=device) * args.guidance_param
            sample = sample_3d_fn(
                model_3d,
                (1, model_3d.njoints, model_3d.nfeats, args.num_frames),
                clip_denoised=False,
                model_kwargs=condition,
                skip_timesteps=0,  # 0
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sample = sample.permute(0, 3, 1, 2).cpu()
            sample = sample.reshape(-1, 23, 3)

            if cur_frame == 60:
                sample_3ds.append(sample[:30])
                temp = sample[30:]
            else:
                sample_3ds.append((sample[:30] * config.lerp_coef.unsqueeze(-1).unsqueeze(-1) + temp * (1 - config.lerp_coef.unsqueeze(-1).unsqueeze(-1))))
                temp = sample[30:]
            if cur_frame == len(orics) and len(orics) % 30 == 0:
                sample_3ds.append(temp)
                break
            elif cur_frame + 30 > len(orics):
                if cur_frame != len(orics):
                    cur_frame = len(orics) - 30
                else:
                    tmp = torch.zeros(30, 23, 3)
                    tmp[:(len(orics) % 30)] = sample[-(len(orics) % 30):]
                    sample_3ds.append(tmp)
                    break
        sample_3ds = torch.stack(sample_3ds).to(device).reshape(-1, 23, 3)[:len(orics)]
        j3dcs = torch.matmul(orics[:len(sample_3ds), -1].unsqueeze(1), sample_3ds.unsqueeze(-1)).squeeze(-1)

        # sample pose and tran
        cur_frame = 0
        temp, temp_tran = [], []
        while cur_frame < len(orics):
            if cur_frame == 0:
                cur_frame += 60
            else:
                cur_frame += 30
            now = time.time()
            condition = {}
            condition['y'] = {}
            condition['y']['scale'] = torch.ones(1, device=device) * args.guidance_param
            condition['y']['3d_imu'] = torch.cat((sample_3ds[cur_frame - 60:cur_frame].flatten(1), orirs[cur_frame - 60:cur_frame].flatten(1), accrs[cur_frame - 60:cur_frame].flatten(1)), dim=1).reshape(1, -1)
            sample = sample_pt_fn(
                model_pose,
                (1, model_pose.njoints, model_pose.nfeats, args.num_frames),
                clip_denoised=False,
                model_kwargs=condition,
                skip_timesteps=0,  # 0
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sample_pose = art.math.r6d_to_rotation_matrix(sample.permute(0, 3, 1, 2).cpu().reshape(-1, 24, 6)).reshape(-1, 24, 3, 3)
            sample_pose = config.body_model.inverse_kinematics_R(sample_pose)
            sample_pose[:, 0] = orics[cur_frame - 60:cur_frame].reshape(-1, 6, 3, 3)[:, -1]
            sample_pose = art.math.axis_angle_to_quaternion(art.math.rotation_matrix_to_axis_angle(sample_pose)).reshape(-1, 24, 4) # for calculating mean rotation
            if cur_frame == 60:
                sample_poses.append(sample_pose[:30])
                temp = sample_pose[30:]
            else:
                temp2 = []
                for i in range(temp.shape[0]):
                    for j in range(temp[i].shape[0]):
                        lerpa = art.math.lerp(temp[i][j], sample_pose[i][j], config.lerp_coef[i])
                        lerpb = art.math.lerp(-temp[i][j], sample_pose[i][j], config.lerp_coef[i])
                        lerpr = lerpa if torch.norm(lerpa) > torch.norm(lerpb) else lerpb
                        temp2.append(lerpr)
                temp2 = torch.stack(temp2).reshape(30, 24, 4)
                sample_poses.append(temp2)
                temp = sample_pose[30:]
            if cur_frame + 30 >= len(orics):
                break

        # pose, tran = art.math.quaternion_to_rotation_matrix(torch.stack(sample_poses)).reshape(-1, 24, 3, 3), torch.stack(sample_trans).reshape(-1, 3)
        # pose, tran, update = smplify_runner(pose[:len(tran)], tran, j2dc_mp_origins[:len(tran)], orics[:len(tran)], batch_size=tran.shape[0], lr=0.001, use_lbfgs=True, opt_steps=1, cam_k=cam_k)
        pose_p.append(art.math.quaternion_to_rotation_matrix(torch.stack(sample_poses)).reshape(-1, 24, 3, 3))
        config.body_model_cpu.view_motion([pose_p[-1]])
        # tran_p.append(torch.stack(sample_trans).reshape(-1, 3))
        # pose_p.append(pose.cpu())
        # tran_p.append(tran.cpu())
        if debug_name is not None:
            path = debug_path
            _vis_overlay(path, pose_p[-1], tran_p[-1], cam_k, debug_name)
    return pose_p, tran_p, pose_t, tran_t


def eval_aist(pt_path='data/results/aist_pose_opt.pt'):
    valid, seq = [], 0
    not_aligned = set([_.strip('\n') for _ in open(os.path.join(config.paths.aist_dir, 'not_aligned.txt')).readlines()])
    dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
    for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
        for j in range(9):  # jth camera view
            cam_name = 'c0' + str(j + 1)
            if dataset['name'][i].replace('cAll', cam_name) not in not_aligned:
                valid.append(seq)
            seq += 1
    if not os.path.exists(pt_path):
        model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d')
        model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond')
        # model_tran, diffusion_tran, _ = _load_model(train_type='diffusion_tran_3dcond')
        dataset = AISTDataset(config.paths.aist_dir, 'test', split_size=-1)
        data = DataLoader(dataset, 1, shuffle=False)
        dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
        pose_p, tran_p, pose_t, tran_t = _eval_ds(dataset, data, model_3d=model_3d, model_tran=None, model_pose=model_pose, diffusion_3d=diffusion_3d, diffusion_pose=diffusion_pose, args=args)
        torch.save((pose_p, tran_p, pose_t, tran_t), pt_path)
    else:
        pose_p, tran_p, pose_t, tran_t = torch.load(pt_path)
    if not os.path.exists(pt_path.replace('.pt', '_error.pt')):
        tran_p = torch.load(os.path.join(config.paths.aist_dir, 'aist_tran.pt'))
        dataset = AISTDataset(config.paths.aist_dir, 'test', split_size=-1)
        data = DataLoader(dataset, 1, shuffle=False)
        dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
        pose_p2 = []
        i = -1
        for item in tqdm.tqdm(data):
            i += 1
            cam_k = dataset['cam_K'][i // 9][i % 9]
            orics = item['oric'].squeeze(0).to(device)
            j2dc_mp_origins = item['j2dc_mp_origin'].squeeze(0)
            pose, _, _ = smplify_runner(pose_p[i], tran_p[i][:len(pose_p[i])], j2dc_mp_origins[:len(pose_p[i])], orics[:len(pose_p[i])], batch_size=pose_p[i].shape[0], lr=0.001, use_lbfgs=True,
                                                opt_steps=1, cam_k=cam_k)
            pose_p2.append(pose)
        torch.save((pose_p2, tran_p, pose_t, tran_t), pt_path.replace('.pt', '_opt.pt'))
        errors = torch.stack([cal_mpjpe(pose_p2[i], pose_t[i][:len(pose_p[i])], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
        torch.save(errors, pt_path.replace('.pt', '_error.pt'))
    else:
        errors = torch.load(pt_path.replace('.pt', '_error.pt'))
    print('mpjpe, pve, pampjpe:', errors[valid].mean(dim=0))
    eval_fn = art.FullMotionEvaluator(config.paths.smpl_file, device=device)
    errors = torch.stack([eval_fn(pose_p[i], pose_t[i][:len(pose_p[i])]) for i in tqdm.trange(len(pose_t))])[valid]
    print('mpjpe, pve, pmpjpe:', errors.mean(dim=0))


def eval_tc(pt_path='data/results/tc_pose111111111111.pt'):
    if not os.path.exists(pt_path):
        model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d')
        model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond')
        # model_tran, diffusion_tran, _ = _load_model(train_type='diffusion_tran_3dcond')
        dataset = TotalCaptureDataset(config.paths.totalcapture_dir, 'test', split_size=-1)
        data = DataLoader(dataset, 1, shuffle=False)
        dataset = torch.load(os.path.join(config.paths.totalcapture_dir, 'test.pt'))
        pose_p, tran_p, pose_t, tran_t = _eval_ds(dataset, data, model_3d=model_3d, model_tran=None, model_pose=model_pose, diffusion_3d=diffusion_3d, diffusion_pose=diffusion_pose, args=args, cam_num=8)
        torch.save((pose_p, tran_p, pose_t, tran_t), pt_path)
    else:
        pose_p, tran_p, pose_t, tran_t = torch.load(pt_path)
    if not os.path.exists(pt_path.replace('.pt', '_error.pt')):
        tran_p = torch.load(os.path.join(config.paths.totalcapture_dir, 'tc_tran.pt'))
        dataset = TotalCaptureDataset(config.paths.totalcapture_dir, 'test', split_size=-1)
        data = DataLoader(dataset, 1, shuffle=False)
        dataset = torch.load(os.path.join(config.paths.totalcapture_dir, 'test.pt'))
        pose_p2 = []
        i = -1
        for item in tqdm.tqdm(data):
            i += 1
            cam_k = dataset['cam_K'][i // 8][i % 8]
            orics = item['oric'].squeeze(0).to(device)
            j2dc_mp_origins = item['j2dc_mp_origin'].squeeze(0)
            pose, _, _ = smplify_runner(pose_p[i], tran_p[i][:len(pose_p[i])], j2dc_mp_origins[:len(pose_p[i])], orics[:len(pose_p[i])], batch_size=pose_p[i].shape[0], lr=0.001, use_lbfgs=True,
                                                opt_steps=1, cam_k=cam_k)
            pose_p2.append(pose.cpu())
        # skip the exception sequence
        pose_p.pop(102)
        pose_t.pop(102)
        errors = torch.stack([cal_mpjpe(pose_p2[i], pose_t[i][:len(pose_p[i])], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
        torch.save(errors, pt_path.replace('.pt', '_error.pt'))
    else:
        errors = torch.load(pt_path.replace('.pt', '_error.pt'))
    print('mpjpe, pve, pampjpe:', errors.mean(dim=0))


def eval_3dpw(occ=False, pt_path='data/results/3dpw_result.pt'):
    model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d')
    model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond')
    sample_3d_fn = diffusion_3d.ddim_sample_loop
    sample_pose_fn = diffusion_pose.ddim_sample_loop
    if not occ:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test', split_size=-1)
    else:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test_occ', split_size=-1)
        pt_path = pt_path.replace('.pt', '_occ.pt')
    _eval_3dpw(dataset, pt_path, args, sample_3d_fn, model_3d, sample_pose_fn, model_pose)

def eval_3dpw_exchange(occ=True, pt_path='data/results/3dpw_result_exchange.pt'):
    model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d')
    model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond')
    sample_3d_fn = diffusion_3d.ddim_sample_loop
    sample_pose_fn = diffusion_pose.ddim_sample_loop
    if not occ:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test', split_size=-1)
    else:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test_occ', split_size=-1)
        pt_path = pt_path.replace('.pt', '_occ.pt')
    _eval_3dpw(dataset, pt_path, args, sample_3d_fn, model_3d, sample_pose_fn, model_pose)

def _eval_3dpw(dataset, pt_path, args, sample_3d_fn, model_3d, sample_pose_fn, model_pose):
    if not os.path.exists(pt_path):
        data = DataLoader(dataset, 1, shuffle=False)
        pose_p, pose_t, idx = [], [], -1
        for item in tqdm.tqdm(data):
            idx += 1
            orirs = item['orir'].squeeze(0).to(device)
            orics = item['oric'].squeeze(0).to(device)
            accrs = item['accr'].squeeze(0).to(device)
            j2ds = item['j2dc_mp2'].squeeze(0).to(device)
            posec = config.body_model_cpu.inverse_kinematics_R(art.math.r6d_to_rotation_matrix(item['posec']).reshape(-1, 24, 3, 3))
            pose_t.append(posec)
            cur_frame = 0
            sample_3ds, temp, sample_poses = [], [], []
            while cur_frame < len(orics):
                if cur_frame == 0:
                    cur_frame += 60
                else:
                    cur_frame += 30
                condition = {}
                condition['y'] = {}
                condition['y']['mp_imu'] = torch.cat((j2ds[cur_frame - 60:cur_frame].flatten(1), orirs[cur_frame - 60:cur_frame].flatten(1), accrs[cur_frame - 60:cur_frame].flatten(1)), dim=1).reshape(1, -1)
                condition['y']['mp'] = j2ds[cur_frame - 60:cur_frame].reshape(1, -1)
                condition['y']['imu'] = torch.cat((orirs[cur_frame - 60:cur_frame].flatten(1), accrs[cur_frame - 60:cur_frame].flatten(1)), dim=1).reshape(1, -1)
                condition['y']['scale'] = torch.ones(1, device=device) * args.guidance_param
                sample = sample_3d_fn(
                    model_3d,
                    (1, model_3d.njoints, model_3d.nfeats, args.num_frames),
                    clip_denoised=False,
                    model_kwargs=condition,
                    skip_timesteps=0,
                    init_image=None,
                    progress=False,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )
                sample = sample.permute(0, 3, 1, 2).cpu()  # (bs, nframes, njoints, nfeats)
                sample = sample.reshape(-1, 23, 3)
                if cur_frame == 60:
                    sample_3ds.append(sample[:30])
                    temp = sample[30:]
                else:
                    sample_3ds.append((sample[:30] * config.lerp_coef.unsqueeze(-1).unsqueeze(-1) + temp * (
                                1 - config.lerp_coef.unsqueeze(-1).unsqueeze(-1))))
                    temp = sample[30:]
                if cur_frame == len(orics) and len(orics) % 30 == 0:
                    sample_3ds.append(temp)
                    break
                elif cur_frame + 30 > len(orics):
                    if cur_frame != len(orics):
                        cur_frame = len(orics) - 30
                    else:
                        tmp = torch.zeros(30, 23, 3)
                        tmp[:(len(orics) % 30)] = sample[-(len(orics) % 30):]
                        sample_3ds.append(tmp)
                        break
            sample_3ds = torch.stack(sample_3ds).to(device).reshape(-1, 23, 3)[:len(orics)]

            cur_frame, temp = 0, []
            while cur_frame < len(orics):
                if cur_frame == 0:
                    cur_frame += 60
                else:
                    cur_frame += 30
                condition = {}
                condition['y'] = {}
                condition['y']['scale'] = torch.ones(1, device=device) * args.guidance_param
                condition['y']['3d_imu'] = torch.cat((sample_3ds[cur_frame - 60:cur_frame].flatten(1), orirs[cur_frame - 60:cur_frame].flatten(1), accrs[cur_frame - 60:cur_frame].flatten(1)), dim=1).reshape(1, -1)
                sample = sample_pose_fn(
                    model_pose,
                    (1, model_pose.njoints, model_pose.nfeats, args.num_frames),
                    clip_denoised=False,
                    model_kwargs=condition,
                    skip_timesteps=0,  # 0
                    init_image=None,
                    progress=False,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )
                sample_pose = art.math.r6d_to_rotation_matrix(sample.permute(0, 3, 1, 2).cpu().reshape(-1, 24, 6)).reshape(-1, 24, 3, 3)
                sample_pose = config.body_model.inverse_kinematics_R(sample_pose)
                sample_pose[:, 0] = orics[cur_frame - 60:cur_frame].reshape(-1, 6, 3, 3)[:, -1]
                sample_pose = art.math.axis_angle_to_quaternion(art.math.rotation_matrix_to_axis_angle(sample_pose)).reshape(-1, 24, 4)
                if cur_frame == 60:
                    sample_poses.append(sample_pose[:30])
                    temp = sample_pose[30:]
                else:
                    temp2 = []
                    for i in range(temp.shape[0]):
                        for j in range(temp[i].shape[0]):
                            lerpa = art.math.lerp(temp[i][j], sample_pose[i][j], config.lerp_coef[i])
                            lerpb = art.math.lerp(-temp[i][j], sample_pose[i][j], config.lerp_coef[i])
                            lerpr = lerpa if torch.norm(lerpa) > torch.norm(lerpb) else lerpb
                            temp2.append(lerpr)
                    temp2 = torch.stack(temp2).reshape(30, 24, 4)
                    sample_poses.append(temp2)
                    temp = sample_pose[30:]
                if cur_frame + 30 >= len(orics):
                    break
            pose_p.append(art.math.quaternion_to_rotation_matrix(torch.stack(sample_poses)).reshape(-1, 24, 3, 3))
        torch.save((pose_p, pose_t), pt_path)
    else:
        pose_p, pose_t = torch.load(pt_path)
    errors = torch.stack([cal_mpjpe(pose_p[i], pose_t[i][:len(pose_p[i])], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
    print('mpjpe, pve, pmpjpe:', errors.mean(dim=0))
    eval_fn = art.FullMotionEvaluator(config.paths.smpl_file, device=device)
    errors = torch.stack([eval_fn(pose_p[i], pose_t[i][:len(pose_p[i])]) for i in tqdm.trange(len(pose_t))])
    print('mpjpe, pve, pmpjpe:', errors.mean(dim=0))


def eval_3dpw_wo_cond(pt_path, occ=False):
    args = edit_args()
    args.model_path = 'checkpoints/diffusion_3d_wo_cond/model002000000.pt'
    args.model_path2 = 'checkpoints/diffusion_pose_3dcond/model002000000.pt'
    model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d_wo_cond', args=args)
    model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond', args=args)
    sample_3d_fn = diffusion_3d.ddim_sample_loop
    sample_pose_fn = diffusion_pose.ddim_sample_loop
    if not occ:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test', split_size=-1)
    else:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test_occ', split_size=-1)
        pt_path = pt_path.replace('.pt', '_occ.pt')
    _eval_3dpw(dataset, pt_path, args, sample_3d_fn, model_3d, sample_pose_fn, model_pose)


def eval_3dpw_wo_input(pt_path, occ=False):
    args = edit_args()
    args.model_path = 'checkpoints/diffusion_3d_wo_input1/model002000000.pt'
    args.model_path2 = 'checkpoints/diffusion_pose_3dcond/model002000000.pt'
    model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d_wo_input', args=args)
    model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond', args=args)
    sample_3d_fn = diffusion_3d.ddim_sample_loop
    sample_pose_fn = diffusion_pose.ddim_sample_loop
    if not occ:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test', split_size=-1)
    else:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test_occ', split_size=-1)
        pt_path = pt_path.replace('.pt', '_occ.pt')
    _eval_3dpw(dataset, pt_path, args, sample_3d_fn, model_3d, sample_pose_fn, model_pose)


def eval_aist_wo_cond(pt_path):
    valid, seq = [], 0
    not_aligned = set([_.strip('\n') for _ in open(os.path.join(config.paths.aist_dir, 'not_aligned.txt')).readlines()])
    dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
    for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
        for j in range(9):  # jth camera view
            cam_name = 'c0' + str(j + 1)
            if dataset['name'][i].replace('cAll', cam_name) not in not_aligned:
                valid.append(seq)
            seq += 1
    if not os.path.exists(pt_path):
        args = edit_args()
        args.model_path = 'checkpoints/diffusion_3d_wo_cond/model002000000.pt'
        args.model_path2 = 'checkpoints/diffusion_pose_3dcond/model002000000.pt'
        model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d_wo_cond', args=args)
        model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond', args=args)
        model_tran, diffusion_tran, _ = _load_model(train_type='diffusion_tran_3dcond')
        dataset = AISTDataset(config.paths.aist_dir, 'test', split_size=-1)
        data = DataLoader(dataset, 1, shuffle=False)
        dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
        pose_p, tran_p, pose_t, tran_t = _eval_ds(dataset, data, model_3d=model_3d, model_tran=model_tran, model_pose=model_pose, diffusion_3d=diffusion_3d, diffusion_pose=diffusion_pose, args=args)
        torch.save((pose_p, tran_p, pose_t, tran_t), pt_path)
    else:
        pose_p, tran_p, pose_t, tran_t = torch.load(pt_path)
    eval_fn = art.PositionErrorEvaluator()
    errors = torch.stack([eval_fn(tran_p[i], tran_t[i][:len(tran_p[i])]) for i in tqdm.trange(len(tran_p))])
    error = errors[valid].mean(dim=0)
    print('absolute root position error:', error)
    if not os.path.exists(pt_path.replace('.pt', '_error.pt')):
        tran_p = torch.load(os.path.join(config.paths.aist_dir, 'aist_tran.pt'))
        dataset = AISTDataset(config.paths.aist_dir, 'test', split_size=-1)
        data = DataLoader(dataset, 1, shuffle=False)
        dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
        pose_p2 = []
        i = -1
        for item in tqdm.tqdm(data):
            i += 1
            cam_k = dataset['cam_K'][i // 9][i % 9]
            orics = item['oric'].squeeze(0).to(device)
            j2dc_mp_origins = item['j2dc_mp_origin'].squeeze(0)
            pose, _, _ = smplify_runner(pose_p[i], tran_p[i][:len(pose_p[i])], j2dc_mp_origins[:len(pose_p[i])],
                                        orics[:len(pose_p[i])], batch_size=pose_p[i].shape[0], lr=0.001, use_lbfgs=True,
                                        opt_steps=1, cam_k=cam_k)
            pose_p2.append(pose.cpu())
        torch.save((pose_p2, tran_p, pose_t, tran_t), pt_path.replace('.pt', '_opt.pt'))
        errors = torch.stack(
            [cal_mpjpe(pose_p2[i], pose_t[i][:len(pose_p[i])], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
        torch.save(errors, pt_path.replace('.pt', '_error.pt'))
    else:
        errors = torch.load(pt_path.replace('.pt', '_error.pt'))
    print('mpjpe, pve, pampjpe:', errors[valid].mean(dim=0))


def eval_tc_wo_cond(pt_path):
    if not os.path.exists(pt_path):
        args = edit_args()
        args.model_path = 'checkpoints/diffusion_3d_wo_cond/model002000000.pt'
        args.model_path2 = 'checkpoints/diffusion_pose_3dcond/model002000000.pt'
        model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d_wo_cond', args=args)
        model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond', args=args)
        model_tran, diffusion_tran, _ = _load_model(train_type='diffusion_tran_3dcond')
        dataset = TotalCaptureDataset(config.paths.totalcapture_dir, 'test', split_size=-1)
        data = DataLoader(dataset, 1, shuffle=False)
        dataset = torch.load(os.path.join(config.paths.totalcapture_dir, 'test.pt'))
        pose_p, tran_p, pose_t, tran_t = _eval_ds(dataset, data, model_3d=model_3d, model_tran=model_tran, model_pose=model_pose, diffusion_3d=diffusion_3d, diffusion_pose=diffusion_pose, args=args, cam_num=8)
        torch.save((pose_p, tran_p, pose_t, tran_t), pt_path)
    else:
        pose_p, tran_p, pose_t, tran_t = torch.load(pt_path)
    eval_fn = art.PositionErrorEvaluator()
    for i in range(len(tran_p)):
        tran_p[i] = art.math.svd_rotate(tran_p[i].unsqueeze(0), tran_t[i][:len(tran_p[i])].unsqueeze(0), False, True, False)[-1][0]
    errors = torch.stack([eval_fn(tran_p[i], tran_t[i][:len(tran_p[i])]) for i in tqdm.trange(len(tran_p))])
    error = errors.mean(dim=0)
    print('absolute root position error:', error)
    if not os.path.exists(pt_path.replace('.pt', '_error.pt')):
        errors = torch.stack([cal_mpjpe(pose_p[i], pose_t[i][:len(pose_p[i])], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
        torch.save(errors, pt_path.replace('.pt', '_error.pt'))
    else:
        errors = torch.load(pt_path.replace('.pt', '_error.pt'))
    print('mpjpe, pve, pampjpe:', errors.mean(dim=0))


def eval_aist_wo_input(pt_path):
    valid, seq = [], 0
    not_aligned = set([_.strip('\n') for _ in open(os.path.join(config.paths.aist_dir, 'not_aligned.txt')).readlines()])
    dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
    for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
        for j in range(9):  # jth camera view
            cam_name = 'c0' + str(j + 1)
            if dataset['name'][i].replace('cAll', cam_name) not in not_aligned:
                valid.append(seq)
            seq += 1
    if not os.path.exists(pt_path):
        args = edit_args()
        args.model_path = 'checkpoints/diffusion_3d_wo_input1/model002000000.pt'
        args.model_path2 = 'checkpoints/diffusion_pose_3dcond/model002000000.pt'
        model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d_wo_input', args=args)
        model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond', args=args)
        model_tran, diffusion_tran, _ = _load_model(train_type='diffusion_tran_3dcond')
        dataset = AISTDataset(config.paths.aist_dir, 'test', split_size=-1)
        data = DataLoader(dataset, 1, shuffle=False)
        dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
        pose_p, tran_p, pose_t, tran_t = _eval_ds(dataset, data, model_3d=model_3d, model_tran=model_tran, model_pose=model_pose, diffusion_3d=diffusion_3d, diffusion_pose=diffusion_pose, args=args)
        torch.save((pose_p, tran_p, pose_t, tran_t), pt_path)
    else:
        pose_p, tran_p, pose_t, tran_t = torch.load(pt_path)
    eval_fn = art.PositionErrorEvaluator()
    errors = torch.stack([eval_fn(tran_p[i], tran_t[i][:len(tran_p[i])]) for i in tqdm.trange(len(tran_p))])
    error = errors[valid].mean(dim=0)
    print('absolute root position error:', error)
    if not os.path.exists(pt_path.replace('.pt', '_error.pt')):
        errors = torch.stack([cal_mpjpe(pose_p[i], pose_t[i][:len(tran_p[i])], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
        torch.save(errors, pt_path.replace('.pt', '_error.pt'))
    else:
        errors = torch.load(pt_path.replace('.pt', '_error.pt'))
    print('mpjpe, pve, pampjpe:', errors[valid].mean(dim=0))


def eval_tc_wo_input(pt_path):
    if not os.path.exists(pt_path):
        args = edit_args()
        args.model_path = 'checkpoints/diffusion_3d_wo_input1/model002000000.pt'
        args.model_path2 = 'checkpoints/diffusion_pose_3dcond/model002000000.pt'
        model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d_wo_input', args=args)
        model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond', args=args)
        model_tran, diffusion_tran, _ = _load_model(train_type='diffusion_tran_3dcond')
        dataset = TotalCaptureDataset(config.paths.totalcapture_dir, 'test', split_size=-1)
        data = DataLoader(dataset, 1, shuffle=False)
        dataset = torch.load(os.path.join(config.paths.totalcapture_dir, 'test.pt'))
        pose_p, tran_p, pose_t, tran_t = _eval_ds(dataset, data, model_3d=model_3d, model_tran=model_tran, model_pose=model_pose, diffusion_3d=diffusion_3d, diffusion_pose=diffusion_pose, args=args, cam_num=8)
        torch.save((pose_p, tran_p, pose_t, tran_t), pt_path)
    else:
        pose_p, tran_p, pose_t, tran_t = torch.load(pt_path)
    eval_fn = art.PositionErrorEvaluator()
    for i in range(len(tran_p)):
        tran_p[i] = art.math.svd_rotate(tran_p[i].unsqueeze(0), tran_t[i][:len(tran_p[i])].unsqueeze(0), False, True, False)[-1][0]
    errors = torch.stack([eval_fn(tran_p[i], tran_t[i][:len(tran_p[i])]) for i in tqdm.trange(len(tran_p))])
    error = errors.mean(dim=0)
    print('absolute root position error:', error)
    if not os.path.exists(pt_path.replace('.pt', '_error.pt')):
        errors = torch.stack([cal_mpjpe(pose_p[i], pose_t[i][:len(pose_p[i])], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
        torch.save(errors, pt_path.replace('.pt', '_error.pt'))
    else:
        errors = torch.load(pt_path.replace('.pt', '_error.pt'))
    print('mpjpe, pve, pampjpe:', errors.mean(dim=0))


def eval_aist_robust(pt_path='data/results/aist_pose_robust2.pt'):
    # !!!! remember to change _eval_ds to change the 2D keypoints noise
    valid, seq = [], 0
    not_aligned = set([_.strip('\n') for _ in open(os.path.join(config.paths.aist_dir, 'not_aligned.txt')).readlines()])
    dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
    for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
        for j in range(9):  # jth camera view
            cam_name = 'c0' + str(j + 1)
            if dataset['name'][i].replace('cAll', cam_name) not in not_aligned:
                valid.append(seq)
            seq += 1
    if not os.path.exists(pt_path):
        model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d')
        model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond')
        model_tran, diffusion_tran, _ = _load_model(train_type='diffusion_tran_3dcond')
        dataset = AISTDataset(config.paths.aist_dir, 'test', split_size=-1)
        data = DataLoader(dataset, 1, shuffle=False)
        dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
        pose_p, tran_p, pose_t, tran_t = _eval_ds(dataset, data, model_3d=model_3d, model_tran=model_tran, model_pose=model_pose, diffusion_3d=diffusion_3d, diffusion_pose=diffusion_pose, args=args)
        torch.save((pose_p, tran_p, pose_t, tran_t), pt_path)
    else:
        pose_p, tran_p, pose_t, tran_t = torch.load(pt_path)
    if not os.path.exists(pt_path.replace('.pt', '_error.pt')):
        errors = torch.stack([cal_mpjpe(pose_p[i], pose_t[i][:len(pose_p[i])], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
        torch.save(errors, pt_path.replace('.pt', '_error.pt'))
    else:
        errors = torch.load(pt_path.replace('.pt', '_error.pt'))
    print('mpjpe, pve, pampjpe:', errors[valid].mean(dim=0))


def eval_one_stage(pt_path, occ=False):
    args = edit_args()
    args.model_path = 'checkpoints/diffusion_pose_1step/model002000000.pt'
    model_3d, diffusion_3d, args = _load_model(train_type='diffusion_pose_1step', args=args)
    sample_2d_fn = diffusion_3d.ddim_sample_loop
    if not occ:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test', split_size=-1)
    else:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test_occ', split_size=-1)
        pt_path = pt_path.replace('.pt', '_occ.pt')
    pose_p, pose_t, idx = [], [], -1
    if not os.path.exists(pt_path):
        data = DataLoader(dataset, 1, shuffle=False)
        for item in tqdm.tqdm(data):
            idx += 1
            orirs = item['orir'].squeeze(0).to(device)
            orics = item['oric'].squeeze(0).to(device)
            accrs = item['accr'].squeeze(0).to(device)
            j2ds = item['j2dc_mp2'].squeeze(0).to(device)
            posec = config.body_model_cpu.inverse_kinematics_R(art.math.r6d_to_rotation_matrix(item['posec']).reshape(-1, 24, 3, 3))
            pose_t.append(posec)
            past_frames, cur_frame, samples_frame, samples_pose_frame = [], 0, None, None
            sample_3ds, mps, temp, sample_poses = [], [], [], []
            while cur_frame < len(orics):
                if cur_frame == 0:
                    cur_frame += 60
                else:
                    cur_frame += 30
                # build model input
                j2dc = j2ds[cur_frame - 60:cur_frame].clone()
                j2dc = j2dc[:, :, :2].reshape(1, args.num_frames, 33, 2)
                j2dc = j2dc.permute(0, 2, 3, 1).to(device)
                condition = {}
                condition['y'] = {}
                condition['y']['mp'] = j2ds[cur_frame - 60:cur_frame].reshape(j2dc.shape[0], -1)
                condition['y']['imu'] = torch.cat(
                    (orirs[cur_frame - 60:cur_frame].flatten(1), accrs[cur_frame - 60:cur_frame].flatten(1)),
                    dim=1).reshape(
                    j2dc.shape[0], -1)
                # add CFG scale to batch
                condition['y']['scale'] = torch.ones(j2dc.shape[0], device=device) * args.guidance_param
                # sample
                sample = sample_2d_fn(
                    model_3d,
                    (j2dc.shape[0], model_3d.njoints, model_3d.nfeats, args.num_frames),
                    clip_denoised=False,
                    model_kwargs=condition,
                    skip_timesteps=0,  # 0
                    init_image=None,
                    progress=False,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )
                sample_pose = art.math.r6d_to_rotation_matrix(sample.permute(0, 3, 1, 2).cpu().reshape(-1, 24, 6)).reshape(
                    -1, 24, 3, 3)
                sample_pose = config.body_model.inverse_kinematics_R(sample_pose)
                sample_pose[:, 0] = orics[cur_frame - 60:cur_frame].reshape(-1, 6, 3, 3)[:, -1]
                sample_pose = art.math.axis_angle_to_quaternion(
                    art.math.rotation_matrix_to_axis_angle(sample_pose)).reshape(-1, 24, 4)
                if cur_frame == 60:
                    sample_poses.append(sample_pose[:30])
                    temp = sample_pose[30:]
                else:
                    temp2 = []
                    for i in range(temp.shape[0]):
                        for j in range(temp[i].shape[0]):
                            lerpa = art.math.lerp(temp[i][j], sample_pose[i][j], config.lerp_coef[i])
                            lerpb = art.math.lerp(-temp[i][j], sample_pose[i][j], config.lerp_coef[i])
                            lerpr = lerpa if torch.norm(lerpa) > torch.norm(lerpb) else lerpb
                            temp2.append(lerpr)
                    temp2 = torch.stack(temp2).reshape(30, 24, 4)
                    sample_poses.append(temp2)
                    temp = sample_pose[30:]
                if cur_frame + 30 >= len(orics):
                    break
            pose_p.append(art.math.quaternion_to_rotation_matrix(torch.stack(sample_poses)).reshape(-1, 24, 3, 3))
            torch.save((pose_p, pose_t), pt_path)
    else:
        pose_p, pose_t = torch.load(pt_path)
    errors = torch.stack(
        [cal_mpjpe(pose_p[i], pose_t[i][:len(pose_p[i])], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
    print('mpjpe, pve, pmpjpe:', errors.mean(dim=0))


def eval_one_stage_tc(pt_path):
    args = edit_args()
    args.model_path = 'checkpoints/diffusion_pose_1step/model002000000.pt'
    model_3d, diffusion_3d, args = _load_model(train_type='diffusion_pose_1step', args=args)
    sample_2d_fn = diffusion_3d.ddim_sample_loop
    dataset = TotalCaptureDataset(config.paths.totalcapture_dir, 'test')
    pose_p, pose_t, idx = [], [], -1
    if not os.path.exists(pt_path):
        data = DataLoader(dataset, 1, shuffle=False)
        for item in tqdm.tqdm(data):
            idx += 1
            orirs = item['orir'].squeeze(0).to(device)
            orics = item['oric'].squeeze(0).to(device)
            accrs = item['accr'].squeeze(0).to(device)
            j2ds = item['j2dc_mp2'].squeeze(0).to(device)
            posec = config.body_model_cpu.inverse_kinematics_R(art.math.r6d_to_rotation_matrix(item['posec']).reshape(-1, 24, 3, 3))
            pose_t.append(posec)
            past_frames, cur_frame, samples_frame, samples_pose_frame = [], 0, None, None
            sample_3ds, mps, temp, sample_poses = [], [], [], []
            while cur_frame < len(orics):
                if cur_frame == 0:
                    cur_frame += 60
                else:
                    cur_frame += 30
                # build model input
                j2dc = j2ds[cur_frame - 60:cur_frame].clone()
                j2dc = j2dc[:, :, :2].reshape(1, args.num_frames, 33, 2)
                j2dc = j2dc.permute(0, 2, 3, 1).to(device)
                condition = {}
                condition['y'] = {}
                condition['y']['mp'] = j2ds[cur_frame - 60:cur_frame].reshape(j2dc.shape[0], -1)
                condition['y']['imu'] = torch.cat(
                    (orirs[cur_frame - 60:cur_frame].flatten(1), accrs[cur_frame - 60:cur_frame].flatten(1)),
                    dim=1).reshape(
                    j2dc.shape[0], -1)
                # add CFG scale to batch
                condition['y']['scale'] = torch.ones(j2dc.shape[0], device=device) * args.guidance_param
                # sample
                sample = sample_2d_fn(
                    model_3d,
                    (j2dc.shape[0], model_3d.njoints, model_3d.nfeats, args.num_frames),
                    clip_denoised=False,
                    model_kwargs=condition,
                    skip_timesteps=0,  # 0
                    init_image=None,
                    progress=False,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )
                sample_pose = art.math.r6d_to_rotation_matrix(sample.permute(0, 3, 1, 2).cpu().reshape(-1, 24, 6)).reshape(
                    -1, 24, 3, 3)
                sample_pose = config.body_model.inverse_kinematics_R(sample_pose)
                sample_pose[:, 0] = orics[cur_frame - 60:cur_frame].reshape(-1, 6, 3, 3)[:, -1]
                sample_pose = art.math.axis_angle_to_quaternion(
                    art.math.rotation_matrix_to_axis_angle(sample_pose)).reshape(-1, 24, 4)
                if cur_frame == 60:
                    sample_poses.append(sample_pose[:30])
                    temp = sample_pose[30:]
                else:
                    temp2 = []
                    for i in range(temp.shape[0]):
                        for j in range(temp[i].shape[0]):
                            lerpa = art.math.lerp(temp[i][j], sample_pose[i][j], config.lerp_coef[i])
                            lerpb = art.math.lerp(-temp[i][j], sample_pose[i][j], config.lerp_coef[i])
                            lerpr = lerpa if torch.norm(lerpa) > torch.norm(lerpb) else lerpb
                            temp2.append(lerpr)
                    temp2 = torch.stack(temp2).reshape(30, 24, 4)
                    sample_poses.append(temp2)
                    temp = sample_pose[30:]
                if cur_frame + 30 >= len(orics):
                    break
            pose_p.append(art.math.quaternion_to_rotation_matrix(torch.stack(sample_poses)).reshape(-1, 24, 3, 3))
            torch.save((pose_p, pose_t), pt_path)
    else:
        pose_p, pose_t = torch.load(pt_path)
    errors = torch.stack(
        [cal_mpjpe(pose_p[i], pose_t[i][:len(pose_p[i])], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
    print('mpjpe, pve, pmpjpe:', errors.mean(dim=0))


def eval_one_stage_aist(pt_path):
    valid = []
    not_aligned = set([_.strip('\n') for _ in open(os.path.join(config.paths.aist_dir, 'not_aligned.txt')).readlines()])
    dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
    seq = 0
    for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
        for j in range(9):  # jth camera view
            cam_name = 'c0' + str(j + 1)
            if dataset['name'][i].replace('cAll', cam_name) not in not_aligned:
                valid.append(seq)
            seq += 1
    args = edit_args()
    args.model_path = 'checkpoints/diffusion_pose_1step/model002000000.pt'
    model_3d, diffusion_3d, args = _load_model(train_type='diffusion_pose_1step', args=args)
    sample_2d_fn = diffusion_3d.ddim_sample_loop
    dataset = AISTDataset(config.paths.aist_dir, 'test')
    pose_p, pose_t, idx = [], [], -1
    if not os.path.exists(pt_path):
        data = DataLoader(dataset, 1, shuffle=False)
        for item in tqdm.tqdm(data):
            idx += 1
            orirs = item['orir'].squeeze(0).to(device)
            orics = item['oric'].squeeze(0).to(device)
            accrs = item['accr'].squeeze(0).to(device)
            j2ds = item['j2dc_mp2'].squeeze(0).to(device)
            posec = config.body_model_cpu.inverse_kinematics_R(art.math.r6d_to_rotation_matrix(item['posec']).reshape(-1, 24, 3, 3))
            pose_t.append(posec)
            past_frames, cur_frame, samples_frame, samples_pose_frame = [], 0, None, None
            sample_3ds, mps, temp, sample_poses = [], [], [], []
            while cur_frame < len(orics):
                if cur_frame == 0:
                    cur_frame += 60
                else:
                    cur_frame += 30
                # build model input
                j2dc = j2ds[cur_frame - 60:cur_frame].clone()
                j2dc = j2dc[:, :, :2].reshape(1, args.num_frames, 33, 2)
                j2dc = j2dc.permute(0, 2, 3, 1).to(device)
                condition = {}
                condition['y'] = {}
                condition['y']['mp'] = j2ds[cur_frame - 60:cur_frame].reshape(j2dc.shape[0], -1)
                condition['y']['imu'] = torch.cat(
                    (orirs[cur_frame - 60:cur_frame].flatten(1), accrs[cur_frame - 60:cur_frame].flatten(1)),
                    dim=1).reshape(
                    j2dc.shape[0], -1)
                # add CFG scale to batch
                condition['y']['scale'] = torch.ones(j2dc.shape[0], device=device) * args.guidance_param
                # sample
                sample = sample_2d_fn(
                    model_3d,
                    (j2dc.shape[0], model_3d.njoints, model_3d.nfeats, args.num_frames),
                    clip_denoised=False,
                    model_kwargs=condition,
                    skip_timesteps=0,  # 0
                    init_image=None,
                    progress=False,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )
                sample_pose = art.math.r6d_to_rotation_matrix(sample.permute(0, 3, 1, 2).cpu().reshape(-1, 24, 6)).reshape(
                    -1, 24, 3, 3)
                sample_pose = config.body_model.inverse_kinematics_R(sample_pose)
                sample_pose[:, 0] = orics[cur_frame - 60:cur_frame].reshape(-1, 6, 3, 3)[:, -1]
                sample_pose = art.math.axis_angle_to_quaternion(
                    art.math.rotation_matrix_to_axis_angle(sample_pose)).reshape(-1, 24, 4)
                if cur_frame == 60:
                    sample_poses.append(sample_pose[:30])
                    temp = sample_pose[30:]
                else:
                    temp2 = []
                    for i in range(temp.shape[0]):
                        for j in range(temp[i].shape[0]):
                            lerpa = art.math.lerp(temp[i][j], sample_pose[i][j], config.lerp_coef[i])
                            lerpb = art.math.lerp(-temp[i][j], sample_pose[i][j], config.lerp_coef[i])
                            lerpr = lerpa if torch.norm(lerpa) > torch.norm(lerpb) else lerpb
                            temp2.append(lerpr)
                    temp2 = torch.stack(temp2).reshape(30, 24, 4)
                    sample_poses.append(temp2)
                    temp = sample_pose[30:]
                if cur_frame + 30 >= len(orics):
                    break
            pose_p.append(art.math.quaternion_to_rotation_matrix(torch.stack(sample_poses)).reshape(-1, 24, 3, 3))
            torch.save((pose_p, pose_t), pt_path)
    else:
        pose_p, pose_t = torch.load(pt_path)
    errors = torch.stack(
        [cal_mpjpe(pose_p[i], pose_t[i][:len(pose_p[i])], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
    print('mpjpe, pve, pmpjpe:', errors[valid].mean(dim=0))


def eval_3dpw_steps(occ=True, steps=1, pt_path='data/results/3dpw_pose_steps1.pt'):
    args = edit_args()
    args.timestep_respacing = 'ddim' + str(steps)
    model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d', args=args)
    model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond', args=args)
    sample_3d_fn = diffusion_3d.ddim_sample_loop
    sample_pose_fn = diffusion_pose.ddim_sample_loop
    if not occ:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test', split_size=-1)
    else:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test_occ', split_size=-1)
        pt_path = pt_path.replace('.pt', '_occ.pt')
    _eval_3dpw(dataset, pt_path, args, sample_3d_fn, model_3d, sample_pose_fn, model_pose)


def eval_aist_steps(steps=1, pt_path='data/results/aist_pose_steps1.pt'):
    valid, seq = [], 0
    not_aligned = set([_.strip('\n') for _ in open(os.path.join(config.paths.aist_dir, 'not_aligned.txt')).readlines()])
    dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
    for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
        for j in range(9):  # jth camera view
            cam_name = 'c0' + str(j + 1)
            if dataset['name'][i].replace('cAll', cam_name) not in not_aligned:
                valid.append(seq)
            seq += 1
    if not os.path.exists(pt_path):
        args = edit_args()
        args.timestep_respacing = 'ddim' + str(steps)
        model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d', args=args)
        model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond', args=args)
        model_tran, diffusion_tran, _ = _load_model(train_type='diffusion_tran_3dcond', args=args)
        dataset = AISTDataset(config.paths.aist_dir, 'test', split_size=-1)
        data = DataLoader(dataset, 1, shuffle=False)
        dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
        pose_p, tran_p, pose_t, tran_t = _eval_ds(dataset, data, model_3d=model_3d, model_tran=model_tran, model_pose=model_pose, diffusion_3d=diffusion_3d, diffusion_pose=diffusion_pose, args=args)
        torch.save((pose_p, tran_p, pose_t, tran_t), pt_path)
    else:
        pose_p, tran_p, pose_t, tran_t = torch.load(pt_path)
    if not os.path.exists(pt_path.replace('.pt', '_error.pt')):
        tran_p = torch.load(os.path.join(config.paths.aist_dir, 'aist_tran.pt'))
        dataset = AISTDataset(config.paths.aist_dir, 'test', split_size=-1)
        data = DataLoader(dataset, 1, shuffle=False)
        dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
        pose_p2 = []
        i = -1
        for item in tqdm.tqdm(data):
            i += 1
            cam_k = dataset['cam_K'][i // 9][i % 9]
            orics = item['oric'].squeeze(0).to(device)
            j2dc_mp_origins = item['j2dc_mp_origin'].squeeze(0)
            pose, _, _ = smplify_runner(pose_p[i], tran_p[i][:len(pose_p[i])], j2dc_mp_origins[:len(pose_p[i])], orics[:len(pose_p[i])], batch_size=pose_p[i].shape[0], lr=0.001, use_lbfgs=True,
                                                opt_steps=1, cam_k=cam_k)
            pose_p2.append(pose)
        torch.save((pose_p2, tran_p, pose_t, tran_t), pt_path.replace('.pt', '_opt.pt'))
        errors = torch.stack([cal_mpjpe(pose_p2[i], pose_t[i][:len(pose_p[i])], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
        torch.save(errors, pt_path.replace('.pt', '_error.pt'))
    else:
        errors = torch.load(pt_path.replace('.pt', '_error.pt'))
    print('mpjpe, pve, pampjpe:', errors[valid].mean(dim=0))

def eval_3dpw_wo_diffusion(occ=True, steps=2, pt_path='data/results/3dpw_pose_wo_diffusion2.pt'):
    args = edit_args()
    args.timestep_respacing = 'ddim2'
    model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d_wo_diff', args=args)
    args = edit_args()
    args.timestep_respacing = 'ddim' + str(steps)
    model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond_wo_diff', args=args)
    sample_3d_fn = diffusion_3d.ddim_sample_loop
    sample_pose_fn = diffusion_pose.ddim_sample_loop
    if not occ:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test', split_size=-1)
    else:
        dataset = Pw3dDataset(config.paths.pw3d_dir, 'test_occ', split_size=-1)
        pt_path = pt_path.replace('.pt', '_occ.pt')
    _eval_3dpw(dataset, pt_path, args, sample_3d_fn, model_3d, sample_pose_fn, model_pose)

def eval_aist_wo_diff(pt_path='data/results/aist_pose_wo_diffusion.pt'):
    valid, seq = [], 0
    not_aligned = set([_.strip('\n') for _ in open(os.path.join(config.paths.aist_dir, 'not_aligned.txt')).readlines()])
    dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
    for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
        for j in range(9):  # jth camera view
            cam_name = 'c0' + str(j + 1)
            if dataset['name'][i].replace('cAll', cam_name) not in not_aligned:
                valid.append(seq)
            seq += 1
    if not os.path.exists(pt_path):
        args = edit_args()
        args.timestep_respacing = 'ddim2'
        model_3d, diffusion_3d, _ = _load_model(train_type='diffusion_3d_wo_diff', args=args)
        model_pose, diffusion_pose, args = _load_model(train_type='diffusion_pose_3dcond_wo_diff', args=args)
        dataset = AISTDataset(config.paths.aist_dir, 'test', split_size=-1)
        data = DataLoader(dataset, 1, shuffle=False)
        dataset = torch.load(os.path.join(config.paths.aist_dir, 'test.pt'))
        pose_p, tran_p, pose_t, tran_t = _eval_ds(dataset, data, model_3d=model_3d, model_tran=None, model_pose=model_pose, diffusion_3d=diffusion_3d, diffusion_pose=diffusion_pose, args=args)
        torch.save((pose_p, tran_p, pose_t, tran_t), pt_path)
    else:
        pose_p, tran_p, pose_t, tran_t = torch.load(pt_path)
    if not os.path.exists(pt_path.replace('.pt', '_error.pt')):
        errors = torch.stack([cal_mpjpe(pose_p[i], pose_t[i][:len(pose_p[i])], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
        torch.save(errors, pt_path.replace('.pt', '_error.pt'))
    else:
        errors = torch.load(pt_path.replace('.pt', '_error.pt'))
    print('mpjpe, pve, pampjpe:', errors[valid].mean(dim=0))
    # eval_fn = art.FullMotionEvaluator(config.paths.smpl_file, device=device)
    # errors = torch.stack([eval_fn(pose_p[i], pose_t[i][:len(pose_p[i])]) for i in tqdm.trange(len(pose_t))])[valid]
    # print('mpjpe, pve, pmpjpe:', errors.mean(dim=0))

if __name__ == '__main__':
    # for comparsion
    # eval_aist()
    eval_tc()
    # eval_3dpw(occ=False)
    # eval_3dpw(occ=True)