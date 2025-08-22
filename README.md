# DiffCap

Code for our paper "DiffCap: Diffusion-based Real-time Human Motion Capture using Sparse IMUs and a Monocular Camera".

## Installation

```
conda create -n diffcap python=3.8
conda activate diffcap
pip install -r requirements.txt
```

Install pytorch cuda version from the official [website](https://pytorch.org/).

## Data

### SMPL Files, Pretrained Model and Test Data

- Download SMPL Files from https://drive.google.com/file/d/1wTgUdHx6_BsufPwPEqeyllb3El_9-mfA/view?usp=drive_link and unzip it to model/.
- DOwnload Checkpoints from https://drive.google.com/file/d/1kpTdyxsrws0XUUBmmM8oieAdP3D_-Fw1/view?usp=drive_link and unzip it to ./.
- Download the test data (https://drive.google.com/file/d/1oDnFd8h4mTCSYKD4zEA0AL3b6qUeUtvl/view?usp=drive_link) and place them at `data/`. Note that this data contains the checkpoints of RobustCap, it does not matter.
- For AIST++ evaluation, download the [no aligned files](https://drive.google.com/file/d/12RSdlg1Px0EUgZKybqY-exUJWK9HskAD/view?usp=drive_link) and place it at `data/dataset_work/AIST`.
- For training, download the data (https://cloud.tsinghua.edu.cn/d/d8d71c25f4ba478e975a/) and place them at `data/dataset_work/`.

## Evaluation

We provide the evaluation code for AIST++, TotalCapture, 3DPW and 3DPW-OCC. Call the specific function to test the corresponding dataset.

```
python sample/eval5.py
```

## Visualization

Use the idx in the evaluation code to specify the sample index. 
- Call the specific function 'config.view_motion([pose_p[-1]])' to visualize the corresponding results by open3d.
- Call the specific function '_vis_overlay(path, pose_p[-1], tran_p[-1], cam_k, debug_name)' to visualize and save the overlay results. Note that you need to specify the video path (download from the official website) and the camera intrinsic parameters and translation (from ground-truth).

## Live Demo

We use 6 Noitom IMUs and a monocular webcam. For different hardwares, you may need to modify the code.

- Config the IMU and camera parameters in `live_detector_noitom.py`.
- Calibrate the camera. We give a simple calibration code in `articulate/utils/executables/RGB_camera_calibration.py`. Then copy the camera intrinsic parameters to `live_detector_noitom.py`.
- Run the live detector code `live_detector_noitom.py` and you can see the camera reading.
- Run the Unity scene to render the results. You can write your own code or use the scene from Transpose (https://github.com/Xinyu-Yi/TransPose).
- Run the live server code `live_server.py` to run our networks and send the results to Unity.

## Training
First modify the config file in `utils/parser_util.py`. 'save_dir' is the path to save the checkpoints. 'train_type' can be 'diffusion_3d' and 'diffusion_pose_3dcond' which means the training of Joint Diffusion Model and Pose Diffusion Model.

```
python train/train_main.py
```

## Citation
```
@article{pan2025diffcap,
  title={DiffCap: Diffusion-Based Real-Time Human Motion Capture Using Sparse IMUs and a Monocular Camera},
  author={Pan, Shaohua and Yi, Xinyu and Zhou, Yan and Jian, Weihua and Zhang, Yuan and Wan, Pengfei and Xu, Feng},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2025},
  publisher={IEEE}
}
```
