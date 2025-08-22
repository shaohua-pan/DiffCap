"""
    Adapted from https://github.com/GuyTevet/motion-diffusion-model.
"""
from argparse import ArgumentParser
import argparse
import os
import json


def parse_and_load_from_model(parser):
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')


def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", default='checkpoints/diffusion_3d', type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--train_platform_type", default='TensorboardPlatform',
                       choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=5_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=2000_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--timestep_respacing", default='', type=str, help="Timestep respacing.")
    group.add_argument("--load_run_3d", default=False, type=bool, help="Whether use the results from joint diffusion model or not.")
    group.add_argument("--down_sample", default=False, type=bool, help="Whether to downsample the input.")
    group.add_argument("--train_type", default='diffusion_3d',
                       choices=['2d', '2d_slide', 'pose', 'pose_no_cond', 'pose_acc_cond', 'diffusion_3d2', 'diffusion_3d_slide', 'diffusion_tran_3dcond',
                                'pose_acc_cond_concat', 'dual', 'pose_2d_acc_cond', '2d_mp', '2d_mp_imu', 'diffusion_tran', 'diffusion_tran_pos',
                                '2d_mp_imu_concat', '2d_nocond', 'diffusion_2d', 'diffusion_3d', 'diffusion_pose_3dcond', 'diffusion_pose', 'diffusion_2d_imur',
                                'diffusion_pose_1step', 'diffusion_3d_wo_input', 'diffusion_3d_wo_cond', 'joint_diffusion_model', 'diffusion_3d_exchange_input_cond',
                                'diffusion_3d_wo_diff', 'diffusion_pose_3dcond_wo_diff', 'diffusion_pose_attn', 'diffusion_pose_edit'], type=str,
                       help="Choose type to train the model.")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--model_path2", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--model_path3", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames.")
    group.add_argument("--timestep_respacing", default='ddim5', type=str, help="Timestep respacing.")


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def edit_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    return parse_and_load_from_model(parser)
