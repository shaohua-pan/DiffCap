import time
import torch
# import win32api
import numpy as np
import cv2
from pygame.time import Clock
import utils
from utils.live_demo_sync import SyncIMUCam
from utils.live_demo_noitom import IMUSet
from utils.live_demo_noitom import apose_tpose_calibration
# from utils.live_reading import IMUSet
import socket
import articulate as art
import config
import mediapipe as mp
from utils.kp_utils import *
# define configs
body_model = art.ParametricModel(config.paths.smpl_file)
K = torch.tensor([[623.79949084, 0., 313.69863974], [0., 623.09646347, 236.76807598], [0., 0., 1.]])
# K = torch.tensor([[610.82898002, 0., 634.04923424], [0., 611.15451114, 362.63987679], [0., 0., 1.]])
height, width = 480, 640
cam_id = 0
save_frame = 2 * 0
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
server_ip = '192.168.1.102'
low_lighting = False

def run_detector(height, width):
    imu_set = IMUSet()
    uv_pre, last_frame = None, None
    cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    cam.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    clock = Clock()
    accs, oris, RCMs, uvs = [], [], [], []
    accss, oriss, RCMss, uvss  = [], [], [], []
    count = 0
    count_save = 3600 * 5
    RMI, RSB = apose_tpose_calibration(imu_set)
    # facing the camera
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if low_lighting:
        uv_pre = torch.rand(33, 3)
        uv_pre[..., -1] = 0.
    with mp_pose.Pose(
            min_detection_confidence=0.0,
            min_tracking_confidence=0.0001,
            model_complexity=1) as mp_detector:
        while True:
            clock.tick(60)
            if uv_pre is None and count % 2 != 0:
                continue
            if count % 2 == 0 and not low_lighting:
                _, frame = cam.read()
            else:
                frame = None
            RCM = torch.tensor([[1, 0, 0], [0, -1, 0.], [0, 0, -1]]).t()
            t, RIS, aI, _, _ = imu_set.get()
            ori = RCM.matmul(RMI).matmul(RIS).matmul(RSB)
            acc = aI.mm(RCM.matmul(RMI).t())
            accs.append(acc)
            oris.append(ori)
            RCMs.append(RCM)
            if frame is not None:
                out.write(frame)
            # if save_frame != 0 and count < save_frame:
            #     accs.append(acc)
            #     oris.append(ori)
            #     RCMs.append(RCM)
            #     if frame is not None:
            #         out.write(frame)
            if frame is not None:
                uv = torch.rand(33, 3)
                uv[:, -1] = 0
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
                if not low_lighting:
                    cv2.imshow('frame', frame)
                    cv2.imshow('landmark', image)
                    c = cv2.waitKey(1)
                    if c == ord('r'):
                        imu_set.clear()
                uv[..., :2] = K.inverse().matmul(art.math.append_one(uv[..., :2]).unsqueeze(-1)).squeeze(-1)[..., :2]
                uv_pre = uv.clone()
            else:
                uv = uv_pre.clone()
                if low_lighting:
                    uv[..., -1] = 0.
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
            # if count == save_frame:L
            #     out.release()
            #     torch.save((oris, accs, RCMs, uvs), name_file + '.pt')


if __name__ == '__main__':
    name_file = str(time.time())
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(name_file + '.mp4', fourcc, 30.0, (width, height))
    if save_frame != 0:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(name_file + '.mp4', fourcc, 30.0, (width, height))
    mp_pose = mp.solutions.pose
    run_detector(height, width)
