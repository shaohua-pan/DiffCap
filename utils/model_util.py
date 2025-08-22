from model.diff2d import Diff2d
from model.diffpose import DiffPose
from model.diff2d_temporal import Diff2dTemporal
from diffusion.respace_live import SpacedLiveDiffusion
from model.diff2d_mp_imu import Diff2d_mp_imu
from model.diff2d_mp_imu_concat import Diff2d_mp_imu_concat
from model.diffpose_2d_acc_cond import DiffPose2dAccCond
from model.diffpose_acc_cond_concat import DiffPoseAccCondConcat
from model.diff2d_slide import Diff2dSlide
from model.diff2d_nocond import Diff2d_nocond
from model.diffpose_acc_cond import DiffPoseAccCond
from model.diffpose_no_cond import DiffPoseNoCond
from model.diffusion_2d import Diffusion2d
from model.diffusion_pose import DiffusionPose
from model.dual_diff import DualDiff
from model.diff2d_mp import Diff2d_mp
from model.diffusion_3d import Diffusion3d
from model.diffusion_tran import DiffusionTran
from model.diffusion_tran_pos import DiffusionTranPos
from model.diffusion_pose_3dcond import DiffusionPose3dCond
from model.diffusion_3d_slide import Diffusion3dSlide
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.diffusion_pose_1step import DiffusionPose1Step
from model.diffusion_tran_3dcond import DiffusionTran3dCond
from model.diffusion_3d_wo_input import Diffusion3dWoInput
from model.diffusion_3d_wocond import Diffusion3dWoCond
from model.joint_diffusion_model import JointDiffusionModel
from model.diffusion_3d_exchange_input_cond import Diffusion3dExchangeInputCond
from model.diffusion_3d_wo_diff import Diffusion3dWoDiff
from model.diffusion_pose_3dcond_wo_diff import DiffusionPose3dCondWoDiff
from model.diffusion_pose_attn import DiffusionPoseAttn
import torch
import numpy as np
import random
from model.diffusion_pose_edit import DiffusionPoseEdit


def create_model_and_diffusion(args, train_type='2d', live=False):
    if train_type == '2d':
        model = Diff2d(**get_model_args(args, train_type))
    elif train_type == '2d_mp':
        model = Diff2d_mp(**get_model_args(args, train_type))
    elif train_type == '2d_mp_imu':
        model = Diff2d_mp_imu(**get_model_args(args, train_type))
    elif train_type == '2d_mp_imu_concat':
        model = Diff2d_mp_imu_concat(**get_model_args(args, train_type))
    elif train_type == '2d_nocond':
        model = Diff2d_nocond(**get_model_args(args, train_type))
    elif train_type == 'pose':
        model = DiffPose(**get_model_args(args, train_type))
    elif train_type == 'pose_no_cond':
        model = DiffPoseNoCond(**get_model_args(args, train_type))
    elif train_type == '2d_slide':
        model = Diff2dSlide(**get_model_args(args, train_type))
    elif train_type == 'pose_acc_cond':
        model = DiffPoseAccCond(**get_model_args(args, train_type))
    elif train_type == 'pose_2d_acc_cond':
        model = DiffPose2dAccCond(**get_model_args(args, train_type))
    elif train_type == 'pose_acc_cond_concat':
        model = DiffPoseAccCondConcat(**get_model_args(args, train_type))
    elif train_type == 'dual':
        model = DualDiff(**get_model_args(args, train_type))
    elif train_type == 'diffusion_2d':
        model = Diffusion2d(**get_model_args(args, train_type))
    elif train_type == 'diffusion_3d':
        model = Diffusion3d(**get_model_args(args, train_type))
    elif train_type == 'diffusion_3d_exchange_input_cond':
        model = Diffusion3dExchangeInputCond(**get_model_args(args, train_type))
    elif train_type == 'diffusion_3d_slide':
        model = Diffusion3dSlide(**get_model_args(args, train_type))
    elif train_type == 'diffusion_pose':
        model = DiffusionPose(**get_model_args(args, train_type))
    elif train_type == 'diffusion_pose_3dcond':
        model = DiffusionPose3dCond(**get_model_args(args, train_type))
    elif train_type == 'diffusion_pose_1step':
        model = DiffusionPose1Step(**get_model_args(args, train_type))
    elif train_type == 'diffusion_tran_3dcond':
        model = DiffusionTran3dCond(**get_model_args(args, train_type))
    elif train_type == 'diffusion_tran':
        model = DiffusionTran(**get_model_args(args, train_type))
    elif train_type == 'diffusion_tran_pos':
        model = DiffusionTranPos(**get_model_args(args, train_type))
    elif train_type == 'diffusion_3d_wo_input':
        model = Diffusion3dWoInput(**get_model_args(args, train_type))
    elif train_type == 'diffusion_3d_wo_cond':
        model = Diffusion3dWoCond(**get_model_args(args, train_type))
    elif train_type == 'joint_diffusion_model':
        model = JointDiffusionModel(**get_model_args(args, train_type))
    elif train_type == 'diffusion_3d_wo_diff':
        model = Diffusion3dWoDiff(**get_model_args(args, train_type))
    elif train_type == 'diffusion_pose_3dcond_wo_diff':
        model = DiffusionPose3dCondWoDiff(**get_model_args(args, train_type))
    elif train_type == 'diffusion_pose_attn':
        model = DiffusionPoseAttn(**get_model_args(args, train_type))
    elif train_type == 'diffusion_pose_edit':
        model = DiffusionPoseEdit(**get_model_args(args, train_type))
    else:
        raise NotImplementedError
    if live:
        diffusion = SpacedLiveDiffusion(args, train_type)
    else:
        diffusion = create_gaussian_diffusion(args, train_type)
    return model, diffusion


def get_model_args(args, train_type='2d'):
    if train_type == '2d':
        njoints = 32
        nfeats = 2
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'imu_dim': args.num_frames * 6 * (9 + 3),
        }
    elif train_type == '2d_mp':
        njoints = 32
        nfeats = 2
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'mp_dim': args.num_frames * 32 * 3,
        }
    elif train_type == '2d_mp_imu':
        njoints = 32
        nfeats = 2
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'mp_imu_dim': args.num_frames * (32 * 3 + 6 * (9 + 3)),
        }
    elif train_type == '2d_mp_imu_concat':
        njoints = 32
        nfeats = 2
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'mp_imu_dim': (32 * 3 + 6 * (9 + 3)),
        }
    elif train_type == 'diffusion_2d':
        njoints = 32
        nfeats = 2
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'imu_dim': 6 * (9 + 3), 'mp_dim': 32 * 3 * args.num_frames,
        }
    elif train_type == 'diffusion_3d':
        njoints = 23
        nfeats = 3
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'imu_dim': 6 * (9 + 3), 'mp_dim': 33 * 3 * args.num_frames,
        }
    elif train_type == 'diffusion_3d_wo_diff':
        njoints = 23
        nfeats = 3
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'imu_dim': 6 * (9 + 3), 'mp_dim': 33 * 3 * args.num_frames,
        }
    elif train_type == 'diffusion_3d_exchange_input_cond':
        njoints = 23
        nfeats = 3
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'imu_dim': 6 * (9 + 3) * args.num_frames, 'mp_dim': 33 * 3,
        }
    elif train_type == 'joint_diffusion_model':
        njoints = 23
        nfeats = 3
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'imu_dim': 6 * (9 + 3), 'mp_dim': 33 * 3 * args.num_frames,
        }
    elif train_type == 'diffusion_3d_wo_input':
        njoints = 23
        nfeats = 3
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'mp_imu_dim': (33 * 3 + 6 * (9 + 3)) * args.num_frames,
        }
    elif train_type == 'diffusion_3d_wo_cond':
        njoints = 23
        nfeats = 3
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'imu_dim': 6 * (9 + 3), 'mp_dim': 33 * 3,
        }
    elif train_type == 'diffusion_pose_1step':
        njoints = 24
        nfeats = 6
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'imu_dim': 6 * (9 + 3), 'mp_dim': 33 * 3 * args.num_frames,
        }
    elif train_type == 'diffusion_3d_slide':
        njoints = 23
        nfeats = 3
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'imu_dim': 6 * (9 + 3), 'mp_dim': 33 * 3 * args.num_frames,
        }
    elif train_type == 'diffusion_2d_imur':
        njoints = 32
        nfeats = 2
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'imu_dim': 6 * (9 + 3), 'mp_dim': 32 * 3 * args.num_frames,
        }
    elif train_type == '2d_nocond':
        njoints = 32
        nfeats = 2
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob
        }
    elif train_type == 'pose':
        njoints = 24
        nfeats = 6
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'd2_dim': args.num_frames * 32 * 2,
        }
    elif train_type == 'pose_no_cond':
        njoints = 24
        nfeats = 6
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu", 'cond_mask_prob': args.cond_mask_prob
        }
    elif train_type == 'pose_acc_cond':
        njoints = 24
        nfeats = 6
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu", 'cond_mask_prob': args.cond_mask_prob,
                'acc_dim': args.num_frames * 6 * 3,
        }
    elif train_type == 'pose_acc_cond_concat':
        njoints = 24
        nfeats = 6
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu", 'cond_mask_prob': args.cond_mask_prob,
                'acc_dim': 6 * 3,
        }
    elif train_type == 'pose_2d_acc_cond':
        njoints = 24
        nfeats = 6
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu", 'cond_mask_prob': args.cond_mask_prob,
                'acc_2d_dim': args.num_frames * (6 * 3 + 32 * 2),
        }
    elif train_type == 'diffusion_pose':
        njoints = 24
        nfeats = 6
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu", 'cond_mask_prob': args.cond_mask_prob,
                'imu_2d_dim': 6 * 3 + 6 * 9 + 32 * 2, 'mp_dim': 32 * 3 * args.num_frames,
        }
    elif train_type == 'diffusion_pose_3dcond':
        njoints = 24
        nfeats = 6
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu", 'cond_mask_prob': args.cond_mask_prob,
                'imu_3d_dim': 6 * 3 + 6 * 9 + 23 * 3, 'mp_dim': 32 * 3 * args.num_frames,
        }
    elif train_type == 'diffusion_pose_edit':
        njoints = 1
        nfeats = 33 * 2 + 6 * 3 + 24 * 6
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu", 'cond_mask_prob': args.cond_mask_prob
        }
    elif train_type == 'diffusion_pose_attn':
        njoints = 24
        nfeats = 6
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu", 'cond_mask_prob': args.cond_mask_prob,
                'imu_3d_dim': 6 * 3 + 6 * 9 + 23 * 3, 'mp_dim': 32 * 3 * args.num_frames,
        }
    elif train_type == 'diffusion_pose_3dcond_wo_diff':
        njoints = 24
        nfeats = 6
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu", 'cond_mask_prob': args.cond_mask_prob,
                'imu_3d_dim': 6 * 3 + 6 * 9 + 23 * 3, 'mp_dim': 32 * 3 * args.num_frames,
        }
    elif train_type == 'diffusion_tran_3dcond':
        njoints = 1
        nfeats = 3
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': 64, 'ff_size': 256, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu", 'cond_mask_prob': args.cond_mask_prob,
                'imu_3d_dim': 6 * 3 + 6 * 9 + 23 * 3, 'mp_dim': 32 * 3 * args.num_frames,
        }
    elif train_type == 'diffusion_tran':
        njoints = 1
        nfeats = 3
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': 64, 'ff_size': 256, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu", 'cond_mask_prob': args.cond_mask_prob,
                'imu_3d_dim': 6 * 3 + 6 * 9 + 23 * 3, 'mp_dim': 33 * 3 * args.num_frames,
        }
    elif train_type == 'diffusion_tran_pos':
        njoints = 1
        nfeats = 3
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': 128, 'ff_size': 512, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu", 'cond_mask_prob': args.cond_mask_prob,
                'imu_2d_dim': 6 * 3 + 6 * 9 + 33 * 3, 'mp_dim': 33 * 3 * args.num_frames,
        }
    elif train_type == '2d_slide':
        njoints = 32
        nfeats = 2
        return {'njoints': njoints, 'nfeats': nfeats,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'imu_dim': args.num_frames * 6 * (9 + 3),
        }
    elif train_type == 'dual':
        return {'njoints_smpl': 24, 'njoints_2d': 32, 'nfeats_smpl': 6, 'nfeats_2d': 2,
                'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
                'dropout': 0.1, 'activation': "gelu",
                'cond_mask_prob': args.cond_mask_prob, 'smpl_dim': args.num_frames * 6 * 24, 'd2_dim': args.num_frames * 32 * 2,
        }


def create_gaussian_diffusion(args, train_type='2d'):
    # default params
    predict_xstart = True
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = args.timestep_respacing
    if train_type == 'diffusion_3d_slide':
        timestep_respacing = ''
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]
    use_timesteps = space_timesteps(steps, timestep_respacing)
    slide = False
    if train_type == '2d_slide' or train_type == 'diffusion_3d_slide':
        use_timesteps = [0, 3, 6, 9, 12]
        slide = True
    return SpacedDiffusion(
        use_timesteps=use_timesteps,
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        slide=slide,
    )


def create_live_gaussian_diffusion(args, train_type='2d'):
    # default params
    predict_xstart = True
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = args.timestep_respacing
    if train_type == 'diffusion_3d_slide':
        timestep_respacing = ''
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]
    use_timesteps = space_timesteps(steps, timestep_respacing)
    slide = False
    if train_type == '2d_slide' or train_type == 'diffusion_3d_slide':
        use_timesteps = [0, 3, 6, 9, 12]
        slide = True
    return SpacedLiveDiffusion(
        use_timesteps=use_timesteps,
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        slide=slide,
    )


def collate_fn(x):
    r"""
    [[seq0, label0], [seq1, label1], [seq2, label2]] -> [[seq0, seq1, seq2], [label0, label1, label2]]
    """
    return list(zip(*x))


def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
