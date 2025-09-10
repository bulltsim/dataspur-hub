# DeepLabCut Seed Project

This directory provides a starting point for training a DeepLabCut (DLC) model for DataSpur Phase 1. You can use it to generate an initial configuration for quadruped pose estimation.

## Quickstart

1. **Install DeepLabCut** (preferably in a conda environment):

   ```bash
   conda create -n dlc python=3.10
   conda activate dlc
   pip install 'deeplabcut[gui]'==2.3.9
   ```

2. **Create a new project** using this dataset as a seed:

   ```python
   import deeplabcut as dlc

   video_path = "path/to/your/video.mp4"
   project_path = dlc.create_new_project(
       "dataspur_dlc",  # project name
       "username",      # your name or team
       [video_path],     # list of videos
       copy_videos=True
   )
   ```

3. **Label a set of frames** using the DLC GUI. Focus on the 10‑15 keypoints defined for bulls and horses (e.g., head, neck, shoulders, elbows, wrists, hips, knees, ankles, tail base).

4. **Prepare the training dataset:**

   ```bash
   dlc.extract_frames(project_path, mode='manual')
   dlc.label_frames(project_path)
   dlc.create_training_dataset(project_path)
   ```

5. **Train the network and evaluate performance** on your annotated data using `dlc.train_network` and `dlc.evaluate_network`.

## Dataset

We recommend starting with the *DLC_Horse* dataset (included as a submodule in this repository) and then fine‑tuning on your own rodeo footage. Transfer learning from horse pose data helps the network adapt to bull and horse movements in rodeo settings.
