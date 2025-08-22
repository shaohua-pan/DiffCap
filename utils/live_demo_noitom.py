r"""
    Live demo using Noitom Perception Neuron Lab IMUs.
"""


__all__ = ['IMUSet', 'tpose_calibration', 'walking_calibration', 'test_sensors']


import tqdm
import time
import socket
import torch
from pygame.time import Clock
import articulate as art
import win32api
import os
from config import *
import keyboard
import datetime
from articulate.utils.noitom import *
import winsound
import matplotlib.pyplot as plt
import cv2


class IMUSet:
    g = 9.797

    def __init__(self, udp_port=7777):
        app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        settings.set_calc_data()
        app.set_settings(settings)
        app.open()
        time.sleep(0.5)

        sensors = [None for _ in range(6)]
        evts = []
        while len(evts) == 0:
            evts = app.poll_next_event()
            for evt in evts:
                assert evt.event_type == MCPEventType.SensorModulesUpdated
                sensor_module_handle = evt.event_data.sensor_module_data.sensor_module_handle
                sensor_module = MCPSensorModule(sensor_module_handle)
                sensors[sensor_module.get_id() - 1] = sensor_module

        print('find %d sensors' % len([_ for _ in sensors if _ is not None]))
        self.app = app
        self.sensors = sensors
        self.t = 0

    def get(self):
        t, RIS, aS, wS, mS = self.get_local()
        aI, wI, mI = self.local_to_global(RIS, aS, wS, mS)
        return t, RIS, aI, wI, mI

    def local_to_global(self, RIS, aS, wS, mS):
        aI = RIS.bmm(aS.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., self.g])  # calculate global free acceleration
        wI = RIS.bmm(wS.unsqueeze(-1)).squeeze(-1)  # calculate global angular velocity
        mI = RIS.bmm(mS.unsqueeze(-1)).squeeze(-1)  # calculate global magnetic field
        return aI, wI, mI

    def get_local(self):
        evts = self.app.poll_next_event()
        if len(evts) > 0:
            self.t = evts[0].timestamp
        q, a, w, m = [], [], [], []
        for sensor in self.sensors:
            q.append(sensor.get_posture())
            a.append(sensor.get_accelerated_velocity())
            w.append(sensor.get_angular_velocity())
            m.append(sensor.get_compass_value())

        # assuming g is positive (= 9.8), we need to change left-handed system to right-handed by reversing axis x, y, z
        RIS = art.math.quaternion_to_rotation_matrix(torch.tensor(q))  # rotation is not changed
        wS = art.math.degree_to_radian(torch.tensor(w))
        mS = torch.tensor(m)
        aS = -torch.tensor(a) / 1000 * self.g  # acceleration is reversed
        return self.t, RIS, aS, wS, mS

    def clear(self):
        pass


def tpose_calibration(imu_set):
    c = input('Used cached RMI? [y]/n    (If you choose no, put imu 1 straight (x = Right, y = Forward, z = Down, Left-handed).')
    if c == 'n' or c == 'N':
        imu_set.clear()
        RSI = imu_set.get()[1][0].view(3, 3).t()
        RMI = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).mm(RSI)
        torch.save(RMI, os.path.join(paths.temp_dir, 'RMI.pt'))
    else:
        RMI = torch.load(os.path.join(paths.temp_dir, 'RMI.pt'))
    print(RMI)

    input('Stand straight in T-pose and press enter. The calibration will begin in 3 seconds')
    time.sleep(3)
    imu_set.clear()
    RIS = imu_set.get()[1]
    RSB = RMI.matmul(RIS).transpose(1, 2).matmul(torch.eye(3))
    return RMI, RSB


def apose_tpose_calibration(imu_set):
    input('Stand in A pose for 3 seconds and then change to T-pose (facing the camera). Press enter to start.')
    time.sleep(3)
    imu_set.clear()
    RIS_A = imu_set.get()[1]
    winsound.Beep(440, 600)
    time.sleep(2)
    imu_set.clear()
    RIS_T = imu_set.get()[1]

    yI = torch.tensor([0, 0, -1.])
    zI = art.math.rotation_matrix_to_axis_angle(RIS_A[:2].bmm(RIS_T[:2].transpose(1, 2)))
    zI[0].neg_()
    zI = zI.mean(dim=0)
    xI = art.math.normalize_tensor(yI.cross(zI))
    zI = art.math.normalize_tensor(xI.cross(yI))
    RMI = torch.stack([xI, yI, zI], dim=0)
    RSB = RMI.matmul(RIS_T).transpose(1, 2).matmul(torch.eye(3))
    torch.save(RMI, os.path.join('data/temp', 'RMI.pt'))
    print(RMI)
    return RMI, RSB


def walking_calibration(imu_set, draw_figures=False):
    def wait_for_still_pose(amax=0.2, wmax=0.1):
        winsound.Beep(440, 600)
        time.sleep(1)
        imu_set.clear()
        while True:
            t, R, a, w = imu_set.get()
            if a.norm(dim=1).max() < amax and w.norm(dim=1).max() < wmax:
                return
    def wait_for_moving_pose(amin=0.3, wmin=0.15):
        winsound.Beep(523, 600)
        time.sleep(1)
        imu_set.clear()
        while True:
            t, R, a, w = imu_set.get()
            if a.norm(dim=1).min() > amin and w.norm(dim=1).min() > wmin:
                return
    def fit_line(points, pass_origin=True):   # l(t) = m + td
        m = points.mean(dim=0)
        if pass_origin:
            m = torch.zeros_like(m)
        d = (points - m).svd().V[:, 0]
        return m, d
    def get_cloest_point(m, d, p):  # l(t) = m + td
        t = -(m - p).mm(d.view(3, 1)) / (d * d).sum()
        m = m + t.view(-1, 1) * d
        return m
    def Jr(theta):
        theta = theta.view(-1, 3)
        a, t = art.math.normalize_tensor(theta, return_norm=True)
        return (t.sin() / t).view(-1, 1, 1) * torch.eye(3, device=theta.device).expand(theta.shape[0], 3, 3) \
               + (1 - t.sin() / t).view(-1, 1, 1) * a.unsqueeze(-1).bmm(a.unsqueeze(-2)) \
               - ((1 - t.cos()) / t).view(-1, 1, 1) * art.math.vector_cross_matrix(a)
    def Jrinv(theta):
        theta = theta.view(-1, 3)
        a, t = art.math.normalize_tensor(theta, return_norm=True)
        J = (t / 2 / (t / 2).tan()).view(-1, 1, 1) * torch.eye(3, device=theta.device).expand(theta.shape[0], 3, 3) \
            + (1 - t / 2 / (t / 2).tan()).view(-1, 1, 1) * a.unsqueeze(-1).bmm(a.unsqueeze(-2)) \
            + (t / 2).view(-1, 1, 1) * art.math.vector_cross_matrix(a)
        J[t.view(-1) < 1e-5] = torch.eye(3, device=theta.device).expand(theta.shape[0], 3, 3)[t.view(-1) < 1e-5]
        return J
    def draw(title, thetat, lines=None):
        # 3D plot
        ax = plt.axes(projection='3d')
        for i in range(4):
            ax.scatter3D(thetat[i][:, 0].numpy(), thetat[i][:, 1].numpy(), thetat[i][:, 2].numpy(), label='IMU %d' % i)
        if lines is not None:
            for i in range(4):
                line = torch.stack((lines[i][0] - 1 * lines[i][1], lines[i][0] + 1 * lines[i][1]))
                ax.plot3D(line[:, 0].numpy(), line[:, 1].numpy(), line[:, 2].numpy(), label='IMU %d line' % i)
        plt.title(title + ' (3D)')
        plt.legend()
        plt.show()

        # 2D plot
        ax = plt.axes()
        for i in range(4):
            ax.scatter(thetat[i][:, 0].numpy(), thetat[i][:, 1].numpy(), label='IMU %d' % i)
        plt.title(title + ' (2D)')
        plt.legend()
        plt.show()

    RMB = torch.tensor([
        [[ 0,  1,  0],
         [-1,  0,  0],
         [ 0,  0,  1]],
        [[ 0, -1,  0],
         [ 1,  0,  0],
         [ 0,  0,  1]],
        [[ 1,  0,  0],
         [ 0,  1,  0],
         [ 0,  0,  1]],
        [[ 1,  0,  0],
         [ 0,  1,  0],
         [ 0,  0,  1]],
        [[ 1,  0,  0],
         [ 0,  1,  0],
         [ 0,  0,  1]],
        [[ 1,  0,  0],
         [ 0,  1,  0],
         [ 0,  0,  1]]
    ]).float()
    N = 600   # total frame of walking
    M = 300   # total frame of walking forward
    input('Press enter to start walking calibration.')
    clock = Clock()
    print('Please stand straight')
    wait_for_still_pose()
    R0 = imu_set.get()[1]

    print('Please walk forward and backward in a straight line')
    wait_for_moving_pose()
    Rt, at, wt = torch.empty(N, 6, 3, 3), torch.empty(N, 6, 3), torch.empty(N, 6, 3)
    for i in tqdm.trange(N):
        t, Rt[i], at[i], wt[i] = imu_set.get()
        clock.tick(60)

    print('Please stand straight')
    wait_for_still_pose()
    R1 = imu_set.get()[1]

    # os.makedirs('data/temp/fitting', exist_ok=True)
    # torch.save({'R0': R0, 'Rt': Rt, 'at': at, 'wt': wt, 'R1': R1}, 'data/temp/fitting/data.pt')
    # R0, Rt, at, wt, R1 = torch.load('data/temp/fitting/data1.pt').values()

    # get average I-pose IMU orientation Rm
    Ipose_maxerr_deg = art.math.radian_to_degree(art.math.angle_between(R0, R1).max())
    if Ipose_maxerr_deg > 20:
        print('Calibration failed: You perform different Ipose before/after the walking, with error %.1f deg' % Ipose_maxerr_deg)
        return False, None, None
    q0 = art.math.axis_angle_to_quaternion(art.math.rotation_matrix_to_axis_angle(R0))
    q1 = art.math.axis_angle_to_quaternion(art.math.rotation_matrix_to_axis_angle(R1))
    qm0 = (q0 + q1) / 2
    qm1 = (q0 - q1) / 2
    t = (qm0.norm(dim=1) < qm1.norm(dim=1)).int().view(6, 1)
    qm = art.math.normalize_tensor(art.math.lerp(qm0, qm1, t))
    Rm = art.math.quaternion_to_rotation_matrix(qm)

    # fit curves
    dRt = Rt.matmul(Rm.transpose(1, 2))
    Ipose_err, axis = [], []
    for i in range(4):
        os.makedirs('data/temp/fitting/IMU%d' % i, exist_ok=True)
        dthetat = art.math.rotation_matrix_to_axis_angle(dRt[:, i])
        Jrinvdthetat = Jrinv(dthetat)
        y = dthetat
        print('------------- Optimization for IMU %d -------------' % i)
        for j in range(20):
            m, d = fit_line(y, pass_origin=True)
            p = get_cloest_point(m, d, dthetat)
            s = torch.linalg.lstsq(Jrinvdthetat.view(-1, 3), (p - dthetat).view(-1, 1)).solution
            y = dthetat + Jrinvdthetat.matmul(s).view(-1, 3)
            print('Iter %2d\t Error: %.2f \t Axis: %s\t dTheta: %s' % (j, torch.dist(y, p).item(), d, s.view(-1)))
            if draw_figures:
                plt.figure()
                ax = plt.axes()
                ax.scatter(dthetat[:, 0].numpy(), dthetat[:, 1].numpy(), label='origin')
                ax.scatter((p - y + dthetat)[:, 0].numpy(), (p - y + dthetat)[:, 1].numpy(), label='fitting')
                plt.legend()
                plt.savefig('data/temp/fitting/IMU%d/Iter%d.png' % (i, j))
                im = cv2.imread('data/temp/fitting/IMU%d/Iter%d.png' % (i, j))
                # plt.show()
                cv2.imshow('fitting', im)
                cv2.waitKey(1)
                plt.close()
        Ipose_err.append(s.view(-1))
        axis.append(d)
    if draw_figures:
        cv2.destroyWindow('fitting')

    # correct I-pose
    Rm[:4] = art.math.axis_angle_to_rotation_matrix(torch.stack(Ipose_err)).transpose(1, 2).bmm(Rm[:4])

    # determine forward direction
    up = torch.tensor([0, 0, -1.])
    forward = torch.stack(axis).cross(up.view(1, 3))
    lpf = art.LowPassFilter(a=0.1)
    sign = torch.zeros(M, 4)
    for i in range(M):
        a = lpf(at[i, 2:4])
        an = a.norm(dim=1)
        if an[1] < 2 < an[0]:  # left foot is moving while right foot is still
            sign[i] = (a[0] * forward).sum(dim=1).sign()   # dot product
        if an[0] < 2 < an[1]:  # right foot is moving while left foot is still
            sign[i] = (a[1] * forward).sum(dim=1).sign()   # dot product

    # if the forward direction is correct, we should get several positive responses first before getting negative
    # responses using the kernel below as we assume the person walks forward first, and vice versa
    kernel = torch.tensor([1., 1, 1, 1, 1, -1, -1, -1, -1, -1])
    threshold = 6
    response = torch.nn.functional.conv1d(sign.t().view(4, 1, 300), kernel.view(1, 1, -1)).view(4, -1).t()
    counter = torch.zeros(4).int()
    for r in response:
        unfinished = counter.abs() < 4
        if unfinished.sum() == 0: break
        counter[unfinished] += (r[unfinished] > threshold).int()
        counter[unfinished] -= (r[unfinished] < -threshold).int()
    if (counter.abs() < 4).sum() > 0:
        print('Calibration failed: Failed to detect enough forward steps during walking')
        return False, None, None
    forward = counter.sign().view(4, 1) * art.math.normalize_tensor(forward)
    e = min([forward[0].dot(forward[1]), forward[0].dot(forward[2]), forward[0].dot(forward[3]),
             forward[1].dot(forward[2]), forward[1].dot(forward[3]), forward[2].dot(forward[3])]).acos()
    if e > 3.14 / 4:
        print('Calibration failed: Max error in forward direction: %.1f degrees' % (e / 3.14 * 180))
        return False, None, None
    print('Calibration succeed: Max error in forward direction: %.1f degrees' % (e / 3.14 * 180))

    # todo: magnetic calibration
    # currently, we use the mean forward
    forward = art.math.normalize_tensor(forward.mean(dim=0))
    left = up.cross(forward)
    RMI = torch.stack([left, up, forward])
    RSB = RMI.matmul(Rm).transpose(1, 2).matmul(RMB)
    return True, RMI, RSB


def test_sensors():
    from articulate.utils.bullet import RotationViewer
    from articulate.utils.pygame import StreamingDataViewer
    clock = Clock()
    imu_set = IMUSet()
    with RotationViewer(6) as rviewer, StreamingDataViewer(6, (-10, 10)) as sviewer:
        imu_set.clear()
        while True:
            clock.tick(60)
            t, R, a = imu_set.get()
            rviewer.update_all(R)
            sviewer.plot(a[:, 1])
            print('time: %.3f' % t, '\tacc:', a.norm(dim=1))


if __name__ == '__main__':
    os.makedirs(paths.temp_dir, exist_ok=True)
    os.makedirs(paths.live_record_dir, exist_ok=True)

    is_executable = False
    server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_for_unity.bind(('127.0.0.1', 8888))
    server_for_unity.listen(1)
    print('Server start. Waiting for unity3d to connect.')
    if paths.unity_file != '' and os.path.exists(paths.unity_file):
        win32api.ShellExecute(0, 'open', os.path.abspath(paths.unity_file), '', '', 1)
        is_executable = True
    conn, addr = server_for_unity.accept()

    imu_set = IMUSet()
    RMI, RSB = tpose_calibration(imu_set)
    net = PIP()
    clock = Clock()
    imu_set.clear()
    data = {'RMI': RMI, 'RSB': RSB, 'aM': [], 'RMB': []}

    while True:
        clock.tick(60)
        tframe, RIS, aI, wI = imu_set.get()
        RMB = RMI.matmul(RIS).matmul(RSB)
        aM = aI.mm(RMI.t())
        pose, tran, cj, grf = net.forward_frame(aM.view(1, 6, 3), RMB.view(1, 6, 3, 3), return_grf=True, only_pose=False)[0]
        pose = art.math.rotation_matrix_to_axis_angle(pose).view(-1, 72)
        tran = tran.view(-1, 3)

        # send motion to Unity
        s = ','.join(['%g' % v for v in pose.view(-1)]) + '#' + \
            ','.join(['%g' % v for v in tran.view(-1)]) + '#' + \
            ','.join(['%d' % v for v in cj]) + '#' + \
            (','.join(['%g' % v for v in grf.view(-1)]) if grf is not None else '') + '$'

        try:
            conn.send(s.encode('utf8'))
        except:
            break

        data['aM'].append(aM)
        data['RMB'].append(RMB)

        if is_executable and keyboard.is_pressed('q'):
            break

        print('\rfps: ', clock.get_fps(), end='')

    if is_executable:
        os.system('taskkill /F /IM "%s"' % os.path.basename(paths.unity_file))

    data['aM'] = torch.stack(data['aM'])
    data['RMB'] = torch.stack(data['RMB'])
    torch.save(data, os.path.join(paths.live_record_dir, 'xsens' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.pt'))
    print('\rFinish.')