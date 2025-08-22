import cv2
import numpy as np
import tqdm
import torch
import enum
import config

mp_mask = torch.tensor(config.mp_mask)
J_regressor = torch.from_numpy(np.load(config.paths.j_regressor_dir)).float()

def view_2d_keypoint(keypoints, parent=None, images=None, thickness=None, skeleton=None, fps=60, color=None):
    r"""
    View 2d keypoint sequence in image coordinate frame. Modified from vctoolkit.render_bones_from_uv.

    Notes
    -----
    If num_frame == 1, only show one picture.
    If parent is None, do not render bones.
    If images is None, use 1080p white canvas.
    If thickness is None, use a default value.
    If keypoints in shape [..., 2], render keypoints without confidence.
    If keypoints in shape [..., 3], render confidence using alpha of colors (more transparent, less confident).

    :param keypoints: Tensor [num_frames, num_joints, *] where *=2 for (u, v) and *=3 for (u, v, confidence).
    :param parent: List in length [num_joints]. e.g., [None, 0, 0, 0, 1, 2, 3 ...]
    :param images: Numpy uint8 array that can expand to [num_frame, height, width, 3].
    :param thickness: Thickness for points and lines.
    :param fps: Sequence FPS.
    """
    if len(keypoints[0].shape) == 2:
        keypoints = keypoints[:, None, :, :]
    if images is None:
        images = [np.ones((keypoints.shape[1], 480, 360, 3), dtype=np.uint8) * 255 for _ in range(keypoints.shape[0])]
    if images[0].dtype != np.uint8:
        raise RuntimeError('images must be uint8 type')
    if thickness is None:
        thickness = int(max(round(images[0].shape[1] / 160), 1))
    for i in range(keypoints.shape[0]):
        images[i] = np.broadcast_to(images[i], (keypoints.shape[1], images[i].shape[-3], images[i].shape[-2], 3))
    has_conf = keypoints.shape[-1] == 3
    is_single_frame = len(images[0]) == 1

    if not is_single_frame:
        writer = cv2.VideoWriter('a.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps,
                                 (keypoints.shape[0] * images[0].shape[2], images[0].shape[1]))
    if color is None:
        color = [(0, 255, 0) for _ in range(keypoints.shape[1])]
    for i in tqdm.trange(len(images[0])):
        bgs = []
        for j in range(keypoints.shape[0]):
            bg = images[j][i]
            for uv in keypoints[j][i]:
                conf = float(uv[2]) if has_conf else 1
                fg = cv2.circle(bg.copy(), (int(uv[0]), int(uv[1])), int(thickness * 2), (0, 0, 255), -1)
                bg = cv2.addWeighted(bg, 1 - conf, fg, conf, 0)
            if parent is not None:
                for c, p in enumerate(parent):
                    if p is not None:
                        start = (int(keypoints[j][i][p][0]), int(keypoints[j][i][p][1]))
                        end = (int(keypoints[j][i][c][0]), int(keypoints[j][i][c][1]))
                        conf = min(float(keypoints[j][i][c][2]), float(keypoints[j][i][p][2])) if has_conf else 1
                        fg = cv2.line(bg.copy(), start, end, (255, 0, 0), thickness)
                        bg = cv2.addWeighted(bg, 1 - conf, fg, conf, 0)
            if skeleton is not None:
                for c, p in skeleton:
                    start = (int(keypoints[j][i][p][0]), int(keypoints[j][i][p][1]))
                    end = (int(keypoints[j][i][c][0]), int(keypoints[j][i][c][1]))
                    conf = min(float(keypoints[j][i][c][2]), float(keypoints[j][i][p][2])) if has_conf else 1
                    fg = cv2.line(bg.copy(), start, end, color[p], thickness)
                    bg = cv2.addWeighted(bg, 1 - conf, fg, conf, 0)
            bgs.append(bg)
        bg = np.concatenate(bgs, axis=1)
        cv2.imshow('2d keypoint', bg)
        if is_single_frame:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
            writer.write(bg)
    if not is_single_frame:
        writer.release()
    cv2.destroyWindow('2d keypoint')


def view_2d_keypoint_on_z_1(keypoints, parent=None, thickness=None, skeleton=None, scale=1, fps=60, color=None):
    r"""
    View 2d keypoint sequence on z=1 plane.

    Notes
    -----
    If num_frame == 1, only show one picture.
    If parent is None, do not render bones.
    If thickness is None, use a default value.
    If keypoints in shape [..., 2], render keypoints without confidence.
    If keypoints in shape [..., 3], render confidence using alpha of colors (more transparent, less confident).

    :param keypoints: Tensor [num_seq, num_frames, num_joints, *] where *=2 for (x, y) and *=3 for (x, y, confidence).
    :param parent: List in length [num_joints]. e.g., [None, 0, 0, 0, 1, 2, 3 ...]
    :param thickness: Thickness for points and lines.
    :param scale: Scale of the keypoints.
    :param fps: Sequence FPS.
    """
    f = 500 * scale
    assert isinstance(keypoints, list)
    keypoints = torch.stack(keypoints).clone()
    keypoints[..., 0] = keypoints[..., 0] * f + 360 / 2
    keypoints[..., 1] = keypoints[..., 1] * f + 480 / 2
    view_2d_keypoint(keypoints, parent=parent, thickness=thickness, skeleton=skeleton, fps=fps, color=color)


def view_2d_keypoint_live(keypoints, parent=None, image=None, thickness=None, skeleton=None, scale=0.7, color=None):
    assert keypoints is not None
    if color is None:
        color = [(0, 255, 0) for _ in range(keypoints.shape[1])]
    f = 500 * scale
    keypoints[..., 0] = keypoints[..., 0] * f + 360 / 2
    keypoints[..., 1] = keypoints[..., 1] * f + 480 / 2
    if image is None:
        image = np.ones((480, 360, 3), dtype=np.uint8) * 255
    if thickness is None:
        thickness = int(max(round(image.shape[1] / 160), 1))
    bg = image
    for j in range(keypoints.shape[0]):
        fg = cv2.circle(bg, (int(keypoints[j][0]), int(keypoints[j][1])), int(thickness * 2), (0, 0, 255), -1)
    if parent is not None:
        for c, p in enumerate(parent):
            if p is not None:
                start = (int(keypoints[p][0]), int(keypoints[p][1]))
                end = (int(keypoints[c][0]), int(keypoints[c][1]))
                fg = cv2.line(bg, start, end, (255, 0, 0), thickness)
    if skeleton is not None:
        for c, p in skeleton:
            start = (int(keypoints[p][0]), int(keypoints[p][1]))
            end = (int(keypoints[c][0]), int(keypoints[c][1]))
            fg = cv2.line(bg, start, end, color[p], thickness)
    return fg


def sync_mp3d_from_smpl(vert, joint):
    syn_3d = vert[:, mp_mask].clone()
    syn_3d[:, 11:17] = joint[:, 16:22].clone()
    syn_3d[:, 23:25] = joint[:, 1:3].clone()
    syn_3d[:, 25:27] = joint[:, 4:6].clone()
    syn_3d[:, 27:29] = joint[:, 7:9].clone()
    return syn_3d


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re

def get_mp_relation():
    skeleton = np.array([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])
    return skeleton


def get_bbox_scale(uv):
    r"""
    max(bbox width, bbox height)
    """
    u_max, u_min = uv[..., 0].max(dim=-1).values, uv[..., 0].min(dim=-1).values
    v_max, v_min = uv[..., 1].max(dim=-1).values, uv[..., 1].min(dim=-1).values
    return torch.max(u_max - u_min, v_max - v_min)


class MediaPipeDetector:

    def __init__(self):
        import mediapipe as mp
        self.mp_detector = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def __call__(self, img, draw_skeleton=False):
        with self.mp_detector.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1) as pose:
            height, width, _ = img.shape
            img.flags.writeable = False
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if draw_skeleton:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_detector.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            if results.pose_landmarks is None:
                uv = torch.rand(33, 3)
                uv[:, 2] = 0
            else:
                uv = []
                for i in results.pose_landmarks.landmark:
                    uv.append([i.x*width, i.y*height, i.visibility])
                uv = torch.tensor(uv)
            if draw_skeleton:
                return uv, image
            return uv


def cal_mpjpe(pose, gt_pose, cal_pampjpe=False):
    _, _, gt_vertices = config.body_model_cpu.forward_kinematics(gt_pose.cpu(), calc_mesh=True)
    J_regressor_batch = J_regressor[None, :].expand(gt_vertices.shape[0], -1, -1)
    gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)[:, :14]
    _, _, vertices = config.body_model_cpu.forward_kinematics(pose.cpu(), calc_mesh=True)
    keypoints_3d = torch.matmul(J_regressor_batch, vertices)[:, :14]
    pred_pelvis = keypoints_3d[:, [0], :].clone()
    gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
    keypoints_3d = keypoints_3d - pred_pelvis
    gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
    if cal_pampjpe:
        pampjpe = reconstruction_error(keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy())
        return torch.tensor([(gt_keypoints_3d - keypoints_3d).norm(dim=2).mean(), (gt_vertices - vertices).norm(dim=2).mean(), pampjpe.mean()])
    return torch.tensor([(gt_keypoints_3d - keypoints_3d).norm(dim=2).mean(), (gt_vertices - vertices).norm(dim=2).mean()])

class PoseLandmark(enum.IntEnum):
  """The 33 pose landmarks."""
  NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32

_POSE_LANDMARKS_LEFT = frozenset([
    PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE,
    PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR, PoseLandmark.MOUTH_LEFT,
    PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW,
    PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_THUMB, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE,
    PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX
])

_POSE_LANDMARKS_RIGHT = frozenset([
    PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE,
    PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR,
    PoseLandmark.MOUTH_RIGHT, PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST,
    PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL,
    PoseLandmark.RIGHT_FOOT_INDEX
])

def get_mp_colors():
    left_spec = (0, 138, 255)
    right_spec = (231, 217, 0)
    colors = [None for _ in range(33)]
    for landmark in _POSE_LANDMARKS_LEFT:
        colors[landmark] = left_spec
    for landmark in _POSE_LANDMARKS_RIGHT:
        colors[landmark] = right_spec
    return colors

def mp_to_smpl(mp_kps):
    """
    Transform mp keypoints to smpl joints.
    :param mp_kps: [N, 33, 3]
    :return: [N, 24, 3]
    """
    smpl_joints = torch.zeros((mp_kps.shape[0], 24, 3), device=mp_kps.device)
    smpl_joints[:, 0] = (mp_kps[:, 23] + mp_kps[:, 24]) / 2
    smpl_joints[:, 0, -1] = 0
    smpl_joints[:, 1] = mp_kps[:, 23]
    smpl_joints[:, 2] = mp_kps[:, 24]
    smpl_joints[:, 3, -1] = 0
    smpl_joints[:, 4] = mp_kps[:, 25]
    smpl_joints[:, 5] = mp_kps[:, 26]
    smpl_joints[:, 6, -1] = 0
    smpl_joints[:, 7] = mp_kps[:, 27]
    smpl_joints[:, 8] = mp_kps[:, 28]
    smpl_joints[:, 9, -1] = 0
    smpl_joints[:, 10] = mp_kps[:, 31]
    smpl_joints[:, 11] = mp_kps[:, 32]
    smpl_joints[:, 12, -1] = 0
    smpl_joints[:, 13, -1] = 0
    smpl_joints[:, 14, -1] = 0
    smpl_joints[:, 15] = mp_kps[:, 0]
    smpl_joints[:, 16] = mp_kps[:, 11]
    smpl_joints[:, 17] = mp_kps[:, 12]
    smpl_joints[:, 18] = mp_kps[:, 13]
    smpl_joints[:, 19] = mp_kps[:, 14]
    smpl_joints[:, 20] = mp_kps[:, 15]
    smpl_joints[:, 21] = mp_kps[:, 16]
    smpl_joints[:, 22] = mp_kps[:, 17]
    smpl_joints[:, 23] = mp_kps[:, 18]
    return smpl_joints

