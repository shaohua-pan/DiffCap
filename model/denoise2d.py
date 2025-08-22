import os
from torch.utils.data import DataLoader, ConcatDataset
import articulate as art
from articulate.utils.print import *
from config import *
from articulate.utils.torch import *
from utils.kp_utils import *
from articulate.utils.torch.train import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = art.ParametricModel(paths.smpl_file, device=device)


class Denoise2d(torch.nn.Module):
    def __init__(self):
        super(Denoise2d, self).__init__()
        self.rnn = RNN(input_size=6 * 3 + 6 * 9 + 32 * 3, output_size=32 * 2, hidden_size=256, num_rnn_layer=2, dropout=0.4)


def train_denoise2d():
    r"""
    Use noise2d and imu as input. Output is 2d pose.
    """
    def AISTDataset(data_dir, kind, split_size=-1):
        r"""
        kind in ['train', 'val', 'test']
        """
        print('Reading %s dataset "%s"' % (kind, data_dir))
        dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        data, label = [], []
        for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
            for j in range(9):  # jth camera view
                if dataset['joint2d_mp'][i][j] is None: continue
                Tcw = dataset['cam_T'][i][j]
                Kinv = dataset['cam_K'][i][j].inverse()
                oric = Tcw[:3, :3].matmul(dataset['imu_ori'][i])
                accc = Tcw.matmul(art.math.append_zero(dataset['imu_acc'][i]).unsqueeze(-1)).squeeze(-1)[..., :3]
                j3dc = Tcw.matmul(art.math.append_one(dataset['sync_3d_mp'][i]).unsqueeze(-1)).squeeze(-1)[..., :3]
                j2dc = j3dc / j3dc[..., -1:]
                j2dc[..., :2] = j2dc[..., :2] / (get_bbox_scale(j2dc)).view(-1, 1, 1)
                j2dc = j2dc[..., :2]
                j2dc = j2dc[:, 1:] - j2dc[:, :1]
                j2dc_mp = torch.zeros(len(oric), 33, 3)
                j2dc_mp[..., :2] = dataset['joint2d_mp'][i][j][..., :2]
                j2dc_mp[..., 0] = j2dc_mp[..., 0] * 1920
                j2dc_mp[..., 1] = j2dc_mp[..., 1] * 1080
                j2dc_mp = Kinv.matmul(art.math.append_one(j2dc_mp[..., :2]).unsqueeze(-1)).squeeze(-1)
                j2dc_mp[..., :2] = j2dc_mp[..., :2] / (get_bbox_scale(j2dc_mp)).view(-1, 1, 1)
                j2dc_mp[:, 1:, :2] = j2dc_mp[:, 1:, :2] - j2dc_mp[:, :1, :2]
                j2dc_mp[..., -1] = dataset['joint2d_mp'][i][j][..., -1]
                j2dc_mp = j2dc_mp[:, 1:]

                if dataset['joint2d_occ'][i][j] is not None:
                    j2dc_mp_noise = torch.zeros(len(oric), 33, 3)
                    j2dc_mp_noise[..., :2] = dataset['joint2d_occ'][i][j][..., :2]
                    j2dc_mp_noise[..., 0] = j2dc_mp_noise[..., 0] * 1920
                    j2dc_mp_noise[..., 1] = j2dc_mp_noise[..., 1] * 1080
                    j2dc_mp_noise = Kinv.matmul(art.math.append_one(j2dc_mp_noise[..., :2]).unsqueeze(-1)).squeeze(-1)
                    j2dc_mp_noise[..., :2] = j2dc_mp_noise[..., :2] / (get_bbox_scale(j2dc_mp_noise)).view(-1, 1, 1)
                    j2dc_mp_noise[:, 1:, :2] = j2dc_mp_noise[:, 1:, :2] - j2dc_mp_noise[:, :1, :2]
                    j2dc_mp_noise[..., -1] = dataset['joint2d_occ'][i][j][..., -1]
                    j2dc_mp_noise = j2dc_mp_noise[:, 1:]
                    data.append(torch.cat((j2dc_mp_noise.flatten(1), accc.flatten(1), oric.flatten(1)), dim=1))
                    label.append(j2dc.flatten(1))
                data.append(torch.cat((j2dc_mp.flatten(1), accc.flatten(1), oric.flatten(1)), dim=1))
                label.append(j2dc.flatten(1))
        return RNNDataset(data, label, split_size=split_size, device=device)

    class AMASSDataset(RNNDataset):
        r"""
        kind in ['train', 'val', 'test']
        """
        def __init__(self, data_dir, kind, split_size=-1):
            print('Reading %s dataset "%s"' % (kind, data_dir))
            dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
            data, label = [], []
            for i in tqdm.trange(len(dataset['imu_acc'])):
                accw = dataset['imu_acc'][i]  # N, 5, 3
                oriw = dataset['imu_ori'][i]  # N, 5, 3, 3
                root = dataset['joint3d'][i][0, 0].clone()
                j3dw_mp = dataset['sync_3d_mp'][i] - root  # N, 33, 3
                data.append(torch.cat((accw.flatten(1), oriw.flatten(1)), dim=1)[1:-1])
                label.append(j3dw_mp.flatten(1)[1:-1])
            super(AMASSDataset, self).__init__(data, label, split_size=split_size)

        def __getitem__(self, i):
            data, label = super(AMASSDataset, self).__getitem__(i)
            accw = data[:, :18].view(-1, 6, 3, 1)
            oriw = data[:, 18:].view(-1, 6, 3, 3)
            j3dw = label.view(-1, 33, 3, 1)
            Rwc0 = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1.]])
            Rc0c = art.math.generate_random_rotation_matrix_constrained(n=1, y=(-180, 180), p=(-30, 30), r=(-5, 5))[0]
            Rcw = Rwc0.mm(Rc0c).t()
            accc = Rcw.matmul(accw)
            oric = Rcw.matmul(oriw)
            j3dc_mp = Rcw.matmul(j3dw).squeeze(-1)
            random_tranc = art.math.lerp(torch.tensor([-1, -1, 3.]), torch.tensor([1, 1, 8.]), torch.rand(3))
            random_tranc[2] -= j3dc_mp[..., -1].min()
            j3dc_mp = j3dc_mp + random_tranc
            j2dc = j3dc_mp / j3dc_mp[..., -1:]
            j2dc = j2dc[..., :2]
            p = 1 - 1 / (torch.rand_like(j2dc[..., -1:]) * 9 + 1)
            j2dc_mp = torch.zeros(len(j2dc), 33, 3)
            j2dc_mp[..., :2] = torch.normal(j2dc[..., :2], 0.01 * (1 - p))
            j2dc_mp[..., :2] = j2dc_mp[..., :2] / get_bbox_scale(j2dc_mp).view(-1, 1, 1)
            j2dc_mp[..., -1:] = p
            j2dc[..., :2] = j2dc[..., :2] / (get_bbox_scale(j2dc)).view(-1, 1, 1)
            j2dc = j2dc[:, 1:] - j2dc[:, :1]
            j2dc_mp[:, 1:, :2] = j2dc_mp[:, 1:, :2] - j2dc_mp[:, :1, :2]
            j2dc_mp = j2dc_mp[:, 1:]
            data = torch.cat((j2dc_mp.flatten(1), accc.flatten(1), oric.flatten(1)), dim=1)
            label = j2dc.flatten(1)
            return data.to(device), label.to(device)

    print_yellow('=================== Training ===================')
    rnn_mse_loss_fn = RNNLossWrapper(torch.nn.MSELoss())
    rnn_dist_eval_fn = RNNLossWrapper(torch.nn.MSELoss())
    save_dir = os.path.join(paths.checkpoints_dir, 'denoise2d')
    net = Denoise2d().rnn.to(device)
    optimizer = torch.optim.Adam(net.parameters())
    train_dataloader = DataLoader(ConcatDataset([
        AISTDataset(paths.aist_dir, kind='train', split_size=200),
        AMASSDataset(paths.amass_dir, kind='train', split_size=200)
    ]), 256, shuffle=True, collate_fn=RNNDataset.collate_fn)
    valid_dataloader = DataLoader(ConcatDataset([
        AISTDataset(paths.aist_dir, kind='val'),
        AMASSDataset(paths.amass_dir, kind='val')
    ]), 64, collate_fn=RNNDataset.collate_fn)
    train(net, train_dataloader, valid_dataloader, save_dir, loss_fn=rnn_mse_loss_fn, eval_fn=rnn_dist_eval_fn,
          num_epoch=100, num_iter_between_vald=20, clip_grad_norm=1, load_last_states=True, optimizer=optimizer,
          eval_metric_names=['2d keypoint denoise error (m)'], wandb_project_name='dual-diffusion',
          wandb_watch=True, wandb_name='denoise2d')


if __name__ == '__main__':
    train_denoise2d()
