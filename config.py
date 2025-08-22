import torch
import articulate as art

class paths:
    smpl_file = 'data/smpl/SMPL_MALE.pkl'
    smpl_file_f = 'data/smpl/SMPL_FEMALE.pkl'

    aist_raw_dir = 'D:/_dataset/AIST/'
    aist_dir = 'data/dataset_work/AIST/'
    aist_tip_dir = 'data/dataset_work/AIST/tip'

    amass_raw_dir = 'D:/_dataset/AMASS/'  # e.g., ACCAD/ACCAD/*/*.npz
    amass_dir = 'data/dataset_work/AMASS/'

    totalcapture_raw_dir = 'D:/_dataset/TotalCapture/'
    totalcapture_dir = 'data/dataset_work/TotalCapture/'
    totalcapture_romp_dir = 'data/dataset_work/TotalCapture/romp'
    totalcapture_pip_dir = 'data/dataset_work/TotalCapture/PIP'
    totalcapture_tip_dir = 'data/dataset_work/TotalCapture/tip'
    totalcapture_pare_dir = r'D:\_dataset\TotalCapture\video2\pare'

    pw3d_raw_dir = 'D:/_dataset/3DPW/'
    pw3d_dir = 'data/dataset_work/3DPW/'
    pw3d_tip_dir = 'data/dataset_work/3DPW/tip'
    pw3d_pip_dir = 'data/dataset_work/3DPW/pip'

    offline_dir = 'data/dataset_work/live/'
    live_dir = 'data/sig_asia/live'

    checkpoints_dir = 'checkpoints/'

    occ_dir = 'F:\ShaohuaPan\dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'
    j_regressor_dir = 'data/dataset_work/J_regressor_h36m.npy'


class amass_data:
    train = ['ACCAD', 'BioMotionLab_NTroje', 'BMLhandball', 'BMLmovi', 'CMU', 'DanceDB', 'DFaust67', 'EKUT',
             'Eyes_Japan_Dataset', 'GRAB', 'HUMAN4D', 'KIT', 'MPI_Limits', 'TCD_handMocap', 'TotalCapture']
    val = ['HumanEva', 'MPI_HDM05', 'MPI_mosh', 'SFU', 'SOMA', 'WEIZMANN', 'Transitions_mocap', 'SSM_synced']
    test = ['TotalCapture']


vel_scale = 3
tran_offset = [0, 0.25, 5]
mp_mask = [332, 2809, 2800, 455, 6260, 3634, 3621, 583, 4071, 45, 3557, 1873, 4123, 1652, 5177, 2235, 5670, 2673, 6133, 2319, 5782, 2746, 6191, 3138, 6528, 1176, 4662, 3381, 6727, 3387, 6787, 3226, 6624]
vi_mask = [1961, 5424, 1176, 4662, 411, 3021]
ji_mask = [18, 19, 4, 5, 15, 0]
smpl_tran_offset = [-0.00217368, -0.240789175, 0.028583793] # smpl root offset in mean shape
contact_threshold = 0.7
lerp_coef = torch.tensor([0.033, 0.066, 0.1, 0.133, 0.166, 0.2, 0.233, 0.266, 0.3, 0.333, 0.366, 0.4, 0.433, 0.466, 0.5, 0.533, 0.566, 0.6, 0.633, 0.666, 0.7, 0.733, 0.766, 0.8, 0.833, 0.866, 0.9, 0.933, 0.966, 1.])
conf_range = (0.7, 0.8)
tran_filter_num = 0.05
distrance_threshold = 10
gravity = -0.018

class Pw3d_data:
    pw3d_occluded_sequences = [
        'courtyard_backpack',
        'courtyard_basketball',
        'courtyard_bodyScannerMotions',
        'courtyard_box',
        'courtyard_golf',
        'courtyard_jacket',
        'courtyard_laceShoe',
        'downtown_stairs',
        'flat_guitar',
        'flat_packBags',
        'outdoors_climbing',
        'outdoors_crosscountry',
        'outdoors_fencing',
        'outdoors_freestyle',
        'outdoors_golf',
        'outdoors_parcours',
        'outdoors_slalom',
    ]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = art.ParametricModel(paths.smpl_file, device=device)
body_model_cpu = art.ParametricModel(paths.smpl_file)
