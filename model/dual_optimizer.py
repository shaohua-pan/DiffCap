import time

import articulate as art
import torch
import numpy as np
import jax.numpy as jnp
import jax
import tqdm

# load data
import articulate.math

class DualOptimizer:

    def __init__(self, lam_3d=0., lam_2d=0.4, lam_3d_prev=0., max_iter=30, tol=1e-4, step_size=5e-1, damping=1e-3, max_damping=1):
        model = art.ParametricModel('data/smpl/SMPL_MALE.pkl')
        zero_pose_joint = model._J
        zero_pose_bone = model.joint_position_to_bone_vector(zero_pose_joint.unsqueeze(0)).squeeze(0)
        zero_pose_bone = zero_pose_bone.unsqueeze(-1).numpy().astype(np.float64)
        self.zero_pose_bone = jnp.array(zero_pose_bone)
        self.lambda1 = lam_3d
        self.lambda2 = lam_2d
        self.lambda3 = lam_3d_prev
        self.max_iter = max_iter
        self.step_size = step_size
        self.tol = tol
        self.damping = damping
        self.max_damping = max_damping
        self.loss = jax.jit(self.loss)
        # self.quaternion_to_rotation_matrix_np = jax.jit(self.quaternion_to_rotation_matrix_np)
        self.fk_np = jax.jit(self.fk_np)
        # self.loss3d = jax.jit(self.loss3d)
        # self.loss2d = jax.jit(self.loss2d)
        self.grad_loss = jax.jit(jax.jacobian(self.loss))

    def quaternion_to_rotation_matrix_np(self, q):
        a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
        r = jnp.concatenate((- 2 * c * c - 2 * d * d + 1, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                             2 * b * c + 2 * a * d, - 2 * b * b - 2 * d * d + 1, 2 * c * d - 2 * a * b,
                             2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, - 2 * b * b - 2 * c * c + 1), axis=1)
        return r.reshape(24, 3, 3)

    def fk_np(self, pose, tran):
        # pose: joint orientation w.r.t camera, [24, 4]
        # tran: tran w.r.t camera, [3]

        pose = self.quaternion_to_rotation_matrix_np(pose)

        bone0 = tran.reshape(3, 1)
        bone1 = pose[0] @ self.zero_pose_bone[1]
        bone2 = pose[0] @ self.zero_pose_bone[2]
        bone3 = pose[0] @ self.zero_pose_bone[3]
        bone4 = pose[1] @ self.zero_pose_bone[4]
        bone5 = pose[2] @ self.zero_pose_bone[5]
        bone6 = pose[3] @ self.zero_pose_bone[6]
        bone7 = pose[4] @ self.zero_pose_bone[7]
        bone8 = pose[5] @ self.zero_pose_bone[8]
        bone9 = pose[6] @ self.zero_pose_bone[9]
        bone10 = pose[7] @ self.zero_pose_bone[10]
        bone11 = pose[8] @ self.zero_pose_bone[11]
        bone12 = pose[9] @ self.zero_pose_bone[12]
        bone13 = pose[9] @ self.zero_pose_bone[13]
        bone14 = pose[9] @ self.zero_pose_bone[14]
        bone15 = pose[12] @ self.zero_pose_bone[15]
        bone16 = pose[13] @ self.zero_pose_bone[16]
        bone17 = pose[14] @ self.zero_pose_bone[17]
        bone18 = pose[16] @ self.zero_pose_bone[18]
        bone19 = pose[17] @ self.zero_pose_bone[19]
        bone20 = pose[18] @ self.zero_pose_bone[20]
        bone21 = pose[19] @ self.zero_pose_bone[21]
        bone22 = pose[20] @ self.zero_pose_bone[22]
        bone23 = pose[21] @ self.zero_pose_bone[23]

        joint0 = bone0
        joint1 = joint0 + bone1
        joint2 = joint0 + bone2
        joint3 = joint0 + bone3
        joint4 = joint1 + bone4
        joint5 = joint2 + bone5
        joint6 = joint3 + bone6
        joint7 = joint4 + bone7
        joint8 = joint5 + bone8
        joint9 = joint6 + bone9
        joint10 = joint7 + bone10
        joint11 = joint8 + bone11
        joint12 = joint9 + bone12
        joint13 = joint9 + bone13
        joint14 = joint9 + bone14
        joint15 = joint12 + bone15
        joint16 = joint13 + bone16
        joint17 = joint14 + bone17
        joint18 = joint16 + bone18
        joint19 = joint17 + bone19
        joint20 = joint18 + bone20
        joint21 = joint19 + bone21
        joint22 = joint20 + bone22
        joint23 = joint21 + bone23

        joint = jnp.stack([joint0, joint1, joint2, joint3, joint4, joint5, joint6, joint7, joint8, joint9, joint10,
                           joint11, joint12, joint13, joint14, joint15, joint16, joint17, joint18, joint19, joint20,
                           joint21, joint22, joint23], axis=0)[:, :, 0]

        return joint

    def loss2d(self, j, jgt):
        j = j / (j[:, 2:] + 1e-7)
        e = jgt[:, 2] * jnp.linalg.norm(j[:, :2] - jgt[:, :2], axis=1)
        # e = jgt[:, 2] * (j[:, :2] - jgt[:, :2]).reshape(-1)
        return e

    def loss3d(self, j, jgt):
        e = jnp.linalg.norm(j - jgt, axis=1)
        # e = (j - jgt).reshape(-1)
        return e

    def loss3d_prev(self, j, jgt):
        e = jnp.linalg.norm(j - jgt, axis=1)
        # e = (j - jgt).reshape(-1)
        return e

    def loss(self, x, gt3d, gt2d, gt3d_prev):
        p = x[:96].reshape(24, 4)
        t = x[96:]
        j = self.fk_np(p, t)
        e3d = self.loss3d(j, gt3d)
        e3d_prev = self.loss3d_prev(j, gt3d_prev)
        e2d = self.loss2d(j, gt2d)
        return self.lambda1 * e3d + self.lambda2 * e2d + self.lambda3 * e3d_prev
        # return self.lambda1 * e3d

    def __call__(self, x, y3d, y2d, y3d_prev, return_3d=False):
        """
        optimize the pose and tran.
        :param x: [pose, tran] (96 + 3) in numpy
        :param y3d: (24, 3) in numpy
        :param y2d: (24, 3) in numpy
        :param y3d_prev:
        :return:
        """
        damping = self.damping
        for _ in range(self.max_iter):
            J = self.grad_loss(x, y3d, y2d, y3d_prev)
            e = self.loss(x, y3d, y2d, y3d_prev)
            H = jnp.dot(J.T, J)
            g = jnp.dot(J.T, e)
            # Introduce damping parameter (lambda) for Levenberg-Marquardt
            dx = np.linalg.solve(H + damping * np.eye(x.shape[0]), -g)
            x_new = x + dx * self.step_size
            # normalize quaternion
            x_new[:96] = (x_new[:96].reshape(24, 4) / np.linalg.norm(x_new[:96].reshape(24, 4), axis=1, keepdims=True)).reshape(-1)
            if e.mean() < self.tol:
                break
            # Check if the update improved the objective function
            e_new = self.loss(x_new, y3d, y2d, y3d_prev)
            if np.sum(e_new ** 2) < np.sum(e ** 2):
                # If the update improves the objective function, decrease the damping factor
                damping /= 1.5
                x = x_new
            else:
                # If the update worsens the objective function, increase the damping factor
                damping = min(damping * 1.5, self.max_damping)
        # print(e.mean())
        if return_3d:
            p = x[:96].reshape(24, 4)
            t = x[96:]
            j = self.fk_np(p, t)
            return x, j
        return x


# data = torch.load('data/dataset_work/TotalCapture/test.pt')
# pose = data['pose'][0]
# tran = data['tran'][0] + 1
# pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
# model = art.ParametricModel('data/smpl/SMPL_MALE.pkl')
# glbpose, jointgt = model.forward_kinematics(pose, tran=tran)
# gt2d = jointgt[:, :, :2] / jointgt[:, :, 2:]
#
# result_pose, result_tran, result_j = [], [], []
# x = np.concatenate((np.broadcast_to(np.array([1, 0, 0, 0.]), (24, 4)).reshape(-1), np.zeros(3)))
# opt = DualOptimizer()
#
#
# def quaternion_to_rotation_matrix(q):
#     a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
#     r = jnp.concatenate((- 2 * c * c - 2 * d * d + 1, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
#                          2 * b * c + 2 * a * d, - 2 * b * b - 2 * d * d + 1, 2 * c * d - 2 * a * b,
#                          2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, - 2 * b * b - 2 * c * c + 1), axis=1)
#     return r.reshape(24, 3, 3)
# for jjj in tqdm.trange(1000):
#     y = jointgt[jjj].numpy()
#     y2d = gt2d[jjj].numpy()
#     x = opt(x, y, y2d, y)
#     pose_pred = torch.tensor(np.array(quaternion_to_rotation_matrix(x[:96].reshape(24, 4))))
#     tran_pred = torch.tensor(np.array(x[96:]))
#     result_pose.append(pose_pred)
#     result_tran.append(tran_pred)
#
# result_pose = torch.stack(result_pose)
# result_tran = torch.stack(result_tran)
# pose_pred = model.inverse_kinematics_R(result_pose)
# model.view_motion([pose_pred, pose[:1000]], [result_tran, tran[:1000]])
