import time
import torch
# import win32api
import numpy as np
import cv2
from pygame.time import Clock
import utils
from utils.live_demo_sync import SyncIMUCam
from utils.live_demo_noitom import IMUSet
import socket
import articulate as art
import config
import mediapipe as mp
from utils.kp_utils import *
# define configs
body_model = art.ParametricModel(config.paths.smpl_file)
# K = torch.tensor([[962.09465926, 0., 472.70325838], [0., 960.45287008, 357.28377582], [0., 0., 1.]])
K = torch.tensor([[623.79949084, 0., 313.69863974], [0., 623.09646347, 236.76807598], [0., 0., 1.]])
height, width = 480, 640
cam_id = 0
save_frame = 2 * 0
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
server_ip = '192.168.1.101'
imus_addr = [
    # 'D4:22:CD:00:44:6E',
    # 'D4:22:CD:00:45:E6',
    'D4:22:CD:00:45:EC',
    'D4:22:CD:00:46:0F',
    'D4:22:CD:00:36:80',
    'D4:22:CD:00:36:04',
    'D4:22:CD:00:32:3E',
    'D4:22:CD:00:32:32',
    # 'D4:22:CD:00:35:4E',
    # 'D4:22:CD:00:36:03',
]

def run_detector(height, width):
    # sync_imu_cam = SyncIMUCam(imus_addr=imus_addr,cam_id=cam_id, height=height, width=width)
    sync_imu_cam = IMUSet(cam_id=cam_id, height=height, width=width)
    uv_pre, last_frame = None, None
    sync_imu_cam.clear()
    clock = Clock()
    accs, oris, RCMs, uvs = [], [], [], []
    count = 0
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    with mp_pose.Pose(
            min_detection_confidence=0.0,
            min_tracking_confidence=0.0001,
            model_complexity=0) as mp_detector:
        while True:
            clock.tick(60)
            # t, ori, acc, frame, RCM = sync_imu_cam.get()
            ori, acc, frame, RCM = sync_imu_cam.get()
            if uv_pre is None and frame is None:
                continue
            if save_frame != 0 and count < save_frame:
                accs.append(acc)
                oris.append(ori)
                RCMs.append(RCM)
                if frame is not None:
                    out.write(frame)
            if frame is not None:
                uv = torch.rand(33, 3)
                uv[:, -1] = 0.
                image = frame.copy()
                # image = frame.copy()
                frame.flags.writeable = False
                results = mp_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    uv = []
                    for i in results.pose_landmarks.landmark:
                        uv.append([i.x * frame.shape[1], i.y * frame.shape[0], i.visibility])
                    uv = torch.tensor(uv)
                    cv2.rectangle(frame, (uv[:, 0].min().int().item(), uv[:, 1].min().int().item()),
                                  (uv[:, 0].max().int().item(), uv[:, 1].max().int().item()), (0, 255, 0), 2)
                cv2.imshow('frame', frame)
                cv2.imshow('landmark', image)
                c = cv2.waitKey(1)
                if c == ord('r'):
                    sync_imu_cam.clear()
                uv[..., :2] = K.inverse().matmul(art.math.append_one(uv[..., :2]).unsqueeze(-1)).squeeze(-1)[..., :2]
                uv_pre = uv.clone()
            else:
                uv = uv_pre.clone()
            uvs.append(uv)
            print(clock.get_fps())
            uv, ori, acc, RCM = uv.cpu().numpy(), ori.cpu().numpy(), acc.cpu().numpy(), RCM.cpu().numpy()
            uvss.append(uv)
            oriss.append(ori)
            accss.append(acc)
            count += 1
            # clock.tick()
            if count == 10:
                data = (','.join([str(i) for i in np.stack(uvss).reshape(-1)]) + '#' + ','.join(
                    [str(i) for i in np.stack(oriss).reshape(-1)]) + '#' + ','.join([str(i) for i in np.stack(accss).reshape(-1)]) + '#' + ','.join(
                    [str(i) for i in RCM.reshape(-1)])).encode()

                s.sendto(data, (server_ip, 9999))
                # data = (','.join([str(i) for i in np.stack(uvss)[40:].reshape(-1)]) + '#' + ','.join(
                #     [str(i) for i in np.stack(oriss)[30:].reshape(-1)]) + '#' + ','.join([str(i) for i in np.stack(accss)[30:].reshape(-1)]) + '#' + ','.join(
                #     [str(i) for i in RCM.reshape(-1)])).encode()
                # s.sendto(data, (server_ip, 9999))
                count = 0
                uvss, oriss, accss = [], [], []


if __name__ == '__main__':
    name_file = str(time.time())
    if save_frame != 0:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(name_file + '.mp4', fourcc, 30.0, (width, height))
    mp_pose = mp.solutions.pose
    run_detector(height, width)
