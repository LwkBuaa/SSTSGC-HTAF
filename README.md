# We can publish the code and the weight after the paper is accepted


# Dependencies

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML == 5.4.1 
- torchpack == 0.2.2
- matplotlib, einops, sklearn, tqdm, wandb, h5py
- Run `pip install -e torchlight` 
- pip install timm==0.3.2

# Data Preparation

### Download datasets.

#### There are 3 datasets to download:

- NTU RGB+D 60
- NTU RGB+D 120
- NW-UCLA

#### NTU RGB+D 60 and 120
1. Request dataset here: https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`


#### NW-UCLA
1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition/
2. Download the dataset:
   1. Move `all_sqe` to `./data/NW-UCLA`


### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu60/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu60
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
 
 cd ./data/ntu120
  # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

# Training & Testing
## Training
- NTU RGB+D 60
```
# Example: train SSGCN(joint CoM 21) on NTU RGB+D 60 cross subject with GPU 0
python get_info.py --config ./config/ntu60-sub/SSGCN_joint_21.yaml --device 0
```

- NTU RGB+D 120
```
# Example: train SSGCN(joint CoM 21) on NTU RGB+D 120 cross subject with GPU 0
python get_info.py --config ./config/ntu120-sub/SSGCN_joint_21.yaml --device 0 --num-class 120
```

- NW-UCLA
```
# Example: train SSGCN(joint CoM 21) on NW-UCLA cross view with GPU 0
python get_info.py --config ./config/ucla/SSGCN_joint_21.yaml --device 0 --num-class 10
```

## Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python get_info.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

- To ensemble the results of different modalities, run 
```
# Example: six-way ensemble for NTU-RGB+D 60 cross-subject
python ensemble.py --dataset ntu/xsub --CoM-21 True -CoM-2 True -CoM-1 True
```








