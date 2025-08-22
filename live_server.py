import threading
import time
import torch
import win32api
import numpy as np
import cv2
from pygame.time import Clock
import utils
import socket
import articulate as art
import config
import cv2
from utils.parser_util import edit_args
from utils.model_util import create_model_and_diffusion, fixseed
from model.cfg_sampler import ClassifierFreeSampleModel
from model.contact_rnn import ContactRNN
from model.position_rnn import PositionRNN
from model.velocity_rnn import VelocityRNN
from utils.kp_utils import *
import _thread as thread
from pygame.time import Clock
from queue import Queue
import threading
import select
from utils.kp_utils import *
from articulate.filter import LowPassFilterRotation

# define configs
device = "cuda" if torch.cuda.is_available() else "cpu"
unity_exe = r'C:\Users\thucg\Desktop\live\ci.exe'
server_ip = '192.168.1.101'
server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_for_unity.bind(('127.0.0.1', 8888))
server_for_unity.listen(1)
print('Server start. Waiting for unity3d to connect.')
conn, addr = server_for_unity.accept()
clock = Clock()
slide_size = 10
temp2d = None
udp_window_size = 10 # must set to 10, modified may cause error
model_contact = ContactRNN(ckpt_path='checkpoints/contact/rnn_contact.pt').to(device)
model_position = PositionRNN(ckpt_path='checkpoints/position/rnn_position.pt').to(device)
model_velocity = VelocityRNN(ckpt_path='checkpoints/velocity/rnn_velocity.pt').to(device)
model_contact.eval()
model_position.eval()
model_velocity.eval()
filt = LowPassFilterRotation(a=0.6)
lerp_coef_10 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

def clear(s):
    while True:
        ready = select.select([s], [], [], 0.01)
        if ready[0]:
            s.recv(int(32 * 99999))
        else:
            return

def convert_from_str(x):
    x = x.split(',')
    data = []
    for i in x:
        data.append(float(i))
    return np.asarray(data)

# global pose and tran queue
p, t = Queue(maxsize=3000), Queue(maxsize=3000)
def send_to_unity():
    global p
    if p.qsize() > 0:
        p0 = p.get()
        global t
        t0 = t.get()
        for i in range(len(p0)):
            clock.tick(60)
            print(clock.get_fps())
            pose = p0[i].view(-1)
            pose = filt(art.math.axis_angle_to_rotation_matrix(pose))
            pose = art.math.rotation_matrix_to_axis_angle(pose).view(-1)
            tran = t0[i].view(-1)
            unity_data = ','.join(['%g' % v for v in pose]) + '#' + \
                         ','.join(['%g' % v for v in tran]) + '$'
            conn.send(unity_data.encode('utf8'))

def run_live_demo(model_3d, model_pose, diffusion, args):
    sample_3d_fn = diffusion.ddim_sample_loop
    sample_pose_fn = diffusion.ddim_sample_loop
    clock_main = Clock()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((server_ip, 9999))
    data, addr = s.recvfrom(4000000)
    uv, ori, acc, RCM = data.decode().split('#')
    RCM = torch.from_numpy(convert_from_str(RCM)).reshape(3, 3).float()
    last_window_3d, last_window_pose, last_window_tran, cur_start, floor_y = None, None, None, None, None
    cur_start = torch.zeros(3)
    trans, poses = [], []
    save_pose, save_tran = [], []
    p_thread = None
    first_window = True
    save = False
    clear(s)
    while True:
        uv = torch.rand(60, 33, 3)
        ori = torch.rand(60, 6, 3, 3)
        acc = torch.rand(60, 6, 3)
        if first_window:
            # compose the 50 frames input
            for i in range(5):
                data, addr = s.recvfrom(99999)
                uv1, ori1, acc1, _ = data.decode().split('#')
                uv1, ori1, acc1 = torch.from_numpy(convert_from_str(uv1)).reshape(udp_window_size, 33, 3).float(), torch.from_numpy(convert_from_str(ori1)).reshape(udp_window_size, 6, 3, 3).float(), torch.from_numpy(convert_from_str(acc1)).reshape(udp_window_size, 6, 3).float()
                uv[i*udp_window_size:(i+1)*udp_window_size], ori[i*udp_window_size:(i+1)*udp_window_size], acc[i*udp_window_size:(i+1)*udp_window_size] = uv1, ori1, acc1
            first_window = False
        else:
            # slide window
            uv[:-slide_size], ori[:-slide_size], acc[:-slide_size] = last_uv[slide_size:], last_ori[slide_size:], last_acc[slide_size:]

        # compose 60 frames
        data, addr = s.recvfrom(99999)
        uv1, ori1, acc1, _ = data.decode().split('#')
        _ = torch.from_numpy(convert_from_str(_)).float()
        if _[..., 0] == 0 and _[..., 1] == 0:
            save = True
        uv1, ori1, acc1 = torch.from_numpy(convert_from_str(uv1)).reshape(udp_window_size, 33, 3).float(), torch.from_numpy(convert_from_str(ori1)).reshape(udp_window_size, 6, 3, 3).float(), torch.from_numpy(convert_from_str(acc1)).reshape(udp_window_size, 6, 3).float()
        uv[-slide_size:], ori[-slide_size:], acc[-slide_size:] = uv1, ori1, acc1

        # save history information
        last_uv, last_ori, last_acc = uv.clone(), ori.clone(), acc.clone()

        uv = uv.reshape(-1, 33, 3)
        ori = ori.reshape(-1, 6, 3, 3)
        acc = acc.reshape(-1, 6, 3)
        now = time.time()
        pose, tran, last_window_pose, floor_y, cur_start = diffusion_sample(model_3d, model_pose, args, uv, ori, acc, sample_3d_fn,
                                                             sample_pose_fn, RCM, last_window_pose, last_window_3d,
                                                             cur_start, floor_y)

        if len(trans) <= 3:
            poses.append(pose.clone())
            trans.append(tran)
        if len(trans) > 3 and floor_y is None:
            pfoot = config.body_model_cpu.forward_kinematics(torch.stack(poses).reshape(-1, 24, 3, 3))[1][:, 10:12]
            tranw = torch.matmul(RCM[:3, :3].T, torch.stack(trans).reshape(-1, 3, 1)).reshape(-1, 3)
            pfootw = torch.min(torch.matmul(RCM[:3, :3].T, pfoot.unsqueeze(-1)).squeeze(-1) + tranw.reshape(-1, 1, 3),dim=1).values
            floor_y = pfootw.mean(dim=0)[1]
            floor_y = floor_y.to(device)

        save_pose.append(pose.clone())
        save_tran.append(tran.clone())

        pose[:, 0] = torch.matmul(RCM.T, pose[:, 0])
        pose = art.math.rotation_matrix_to_axis_angle(pose).reshape(-1, 24, 3)
        global p, t
        if p_thread is not None:
            p_thread.join()
        p.put(pose)
        tran = torch.matmul(RCM.T, tran.unsqueeze(-1)).squeeze(-1)
        # tran = torch.zeros(pose.shape[0], 3)
        t.put(tran)
        p_thread = threading.Thread(target=send_to_unity, args=())
        p_thread.start()

        # last_30_pose = pose_30
        # pose[:, 0] = torch.matmul(RCM.T, pose[:, 0])
        # pose = art.math.rotation_matrix_to_axis_angle(pose[-1]).view(24, 3).view(-1)
        # im = view_2d_keypoint_live(kp)
        # cv2.imshow('f', im)
        # cv2.waitKey(1)

        # tran = torch.tensor([0,0,0])
        # unity_data = ','.join(['%g' % v for v in pose]) + '#' + \
        #              ','.join(['%g' % v for v in tran]) + '$'
        # conn.send(unity_data.encode('utf8'))

        #
        # pppp = thread.start_new_thread(send_to_unity, ('', 2))
        # global temp
        # temp = 1
        # f += 14
        # print(time.time()-now)

def diffusion_sample(model_3d, model_pose, args, uv, ori, acc, sample_3d_fn, sample_pose_fn, RCM, last_window_pose,
                     last_window_3d, cur_start, floor_y):
    j2dcs = uv.clone()
    j2dcs[..., :2] = j2dcs[..., :2] / (get_bbox_scale(j2dcs)).view(-1, 1, 1)
    j2dcs[:, 24:, :2] = j2dcs[:, 24:, :2] - j2dcs[:, 23:24, :2]
    j2dcs[:, :23, :2] = j2dcs[:, :23, :2] - j2dcs[:, 23:24, :2]
    # j2dcs[..., -1] = 0.1
    # todo: check thisx
    # if j2dcs[..., 2].mean().item() < 0.8 and temp2d is not None:
        # j2dcs[..., :2] = torch.rand(j2dcs[..., :2].shape) * 0.1
        # j2dcs[..., :2] = temp2d[..., :2]
    # else:
    #     temp_j2dc = j2dcs.clone()
    oric = ori
    accc = acc
    if uv[..., 2].mean().item() < 0.8:
        uv[..., :2] = torch.rand(uv[..., :2].shape)
    uv = uv.to(device)
    j2dcs = j2dcs.to(device)
    Rrc = ori[:, -1].transpose(-1, -2)
    orir = torch.matmul(Rrc.unsqueeze(1), oric).reshape(-1, 6, 3, 3).to(device)
    accr = torch.matmul(Rrc.unsqueeze(1), accc.unsqueeze(-1)).reshape(-1, 6, 3).to(device)

    # joint diffusion model prediction
    condition = {}
    condition['y'] = {}
    condition['y']['mp'] = j2dcs.reshape(1, -1)
    condition['y']['imu'] = torch.cat((orir.flatten(1), accr.flatten(1)), dim=1).reshape(1, -1).to(device)
    condition['y']['scale'] = torch.ones(1, device=device) * args.guidance_param
    sample = sample_3d_fn(model_3d, (1, model_3d.njoints, model_3d.nfeats, args.num_frames), clip_denoised=False, model_kwargs=condition)
    joint = sample.permute(0, 3, 1, 2).reshape(-1, 23, 3) # (60, 23, 3)

    # pose diffusion model prediction
    condition['y']['3d_imu'] = torch.cat((joint.flatten(1), orir.flatten(1), accr.flatten(1)), dim=1).reshape(1, -1)
    sample = sample_pose_fn(model_pose, (1, model_pose.njoints, model_pose.nfeats, args.num_frames), clip_denoised=False, model_kwargs=condition)
    sample_pose = art.math.r6d_to_rotation_matrix(sample.permute(0, 3, 1, 2).cpu().reshape(-1, 24, 6)).reshape(-1, 24, 3, 3)
    sample_pose = config.body_model.inverse_kinematics_R(sample_pose)
    sample_pose[:, 0] = oric.reshape(-1, 6, 3, 3)[:, -1]
    sample_pose = art.math.axis_angle_to_quaternion(art.math.rotation_matrix_to_axis_angle(sample_pose)).reshape(-1, 24, 4)
    if last_window_pose is None:
        pose = sample_pose[-slide_size*2:-slide_size]
        last_window_pose = sample_pose[-slide_size:]
    else:
        pose = []
        for i in range(slide_size): # each frame
            for j in range(24): # each joint rotation
                lerpa = art.math.lerp(last_window_pose[i][j], sample_pose[-slide_size*2:-slide_size][i][j], lerp_coef_10[i])
                lerpb = art.math.lerp(-last_window_pose[i][j], sample_pose[-slide_size*2:-slide_size][i][j], lerp_coef_10[i])
                lerpr = lerpa if torch.norm(lerpa) > torch.norm(lerpb) else lerpb
                pose.append(lerpr)
        pose = torch.stack(pose).reshape(-1, 24, 4) # (slide_size, 24, 4)
        last_window_pose = sample_pose[-slide_size:]
    pose = art.math.quaternion_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
    # pose[:, 0] = oric.reshape(-1, 6, 3, 3)[-slide_size:, -1]
    # return pose, last_window_pose

    # translation estimation
    now = time.time()
    oric, accc = oric.to(device), accc.to(device)
    j3dc = torch.matmul(oric[:, -1].unsqueeze(1), joint.unsqueeze(-1)).squeeze(-1)
    contact = model_contact([torch.cat((accr[-slide_size*2:-slide_size].flatten(1), orir[-slide_size*2:-slide_size].flatten(1), joint[-slide_size*2:-slide_size].flatten(1)), dim=1)])[0].sigmoid()
    contact[0] = 0
    # todo: remeber the wrong hidden state problem
    position = model_position([torch.cat((accc[-slide_size*2:-slide_size].flatten(1), oric[-slide_size*2:-slide_size].flatten(1), uv[-slide_size*2:-slide_size].flatten(1), j3dc[-slide_size*2:-slide_size].flatten(1)), dim=1)])[0]
    velocity = model_velocity([torch.cat((accr[-slide_size*2:-slide_size].flatten(1), oric[-slide_size*2:-slide_size].flatten(1), joint[-slide_size*2:-slide_size].flatten(1)), dim=1)])[0]
    velocity = torch.matmul(oric[-slide_size*2:-slide_size][:, -1], velocity.reshape(-1, 3, 1)).reshape(-1, 3) * config.vel_scale / 60
    tran = cal_tran(position, contact, velocity, pose, floor_y, j2dcs, cur_start, RCM.T)
    cur_start = tran[-1]
    print('time2:' + str(time.time()-now))
    return pose, tran, last_window_pose, floor_y, cur_start


def cal_tran(position, contact, velocity, pose, floor, j2dcs, tstart, Rwc):
    contact, position, velocity = contact[:30].cpu(), position[:30].cpu(), velocity[:30].cpu()
    tstart = tstart.cpu()
    Rwc = Rwc.cpu()
    # use contact to update the translation
    pfoot = config.body_model_cpu.forward_kinematics(pose)[1][:, 10:12]  # (n_frame, 2, 3)
    pfoot_v = torch.zeros_like(pfoot)
    pfoot_v[1:] = pfoot[:-1] - pfoot[1:]  # (n_frame, 2, 3)
    pfoot_v = art.math.lerp(pfoot_v[:, 0], pfoot_v[:, 1], contact.max(dim=1).indices.view(-1, 1))
    weight = (contact.max(dim=1).values.clamp(0.5, 0.9) - 0.5) / 0.4
    tran = velocity * (1 - weight.unsqueeze(-1)) + pfoot_v * weight.unsqueeze(-1)
    # complementary filter of the translation
    position = position.reshape(-1, 3)
    c = j2dcs[..., -1].mean(dim=1)
    tran_lerp = torch.zeros(position.shape).to(position.device)
    for i in range(len(position)):
        if i == 0:
            tranv = tran[0] + tstart
        else:
            tranv = tran_lerp[i - 1] + tran[i]
        k = (c[i] - config.conf_range[0]) / (config.conf_range[1] - config.conf_range[0])
        if k > 1: k = 1
        if (position[i] - tranv).norm() > config.distrance_threshold or config.tran_filter_num >= 1:
            tran_lerpi = position[i]
        elif c[i] >= config.conf_range[1]:
            tran_lerpi = art.math.lerp(tranv, position[i], config.tran_filter_num * k)
        else:
            tran_lerpi = tranv

        # add gravity
        if floor is not None:
            floor = floor.cpu()
            tranw = torch.matmul(Rwc, tran_lerpi.reshape(3, 1)).reshape(3)
            tranw[1] += config.gravity
            pfootw = torch.matmul(Rwc, pfoot.unsqueeze(-1)).squeeze(-1)[i] + tranw.reshape(1, 3)
            down_floor = floor - (pfootw[:, 1].min())
            tranw[1] += torch.max(down_floor, torch.zeros_like(down_floor))
            tran_lerpi = torch.matmul(Rwc.T, tranw.reshape(3, 1)).reshape(3)
        tran_lerp[i] = tran_lerpi
    tran = tran_lerp
    return tran

def _load_model(train_type='2d', args=None):
    if args is None:
        args = edit_args()
    fixseed(args.seed)
    model, diffusion = create_model_and_diffusion(args, train_type)
    if train_type == 'diffusion_3d':
        state_dict = torch.load(args.model_path, map_location='cpu')
    elif train_type == 'diffusion_pose_3dcond':
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


if __name__ == '__main__':
    model_3d, diffusion, _ = _load_model(train_type='diffusion_3d')
    model_pose, _, args = _load_model(train_type='diffusion_pose_3dcond')
    model_tran = None
    run_live_demo(model_3d, model_pose, diffusion, args)
