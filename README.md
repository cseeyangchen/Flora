<div align="center">
<h2>Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: <br>Towards Robust Zero-Shot Skeleton Action Recognition</h2>

<div>    
    <a href='https://cseeyangchen.github.io/' target='_blank'>Yang Chen</a><sup>1</sup>&nbsp&nbsp&nbsp&nbsp;
    <a href='https://keepgoingjkg.github.io/about/' target='_blank'>Miaoge Li</a><sup>1</sup>&nbsp&nbsp&nbsp&nbsp;
    <a href='https://zjrao.github.io/' target='_blank'>Zhijie Rao</a><sup>1</sup>&nbsp&nbsp&nbsp&nbsp;
    <a href='https://dblp.org/pid/08/5937.html' target='_blank'>Deze Zeng</a><sup>2</sup>&nbsp&nbsp&nbsp&nbsp;
    <a href='https://cse.hkust.edu.hk/~songguo/' target='_blank'>Song Guo</a><sup>3</sup>&nbsp&nbsp&nbsp&nbsp;
    <a href='https://jingcaiguo.github.io/' target='_blank'>Jingcai Guo</a><sup>1*</sup>&nbsp&nbsp&nbsp&nbsp;
</div>

<br>

<div> 
PolyU<sup>1</sup>&nbsp&nbsp&nbsp&nbsp;
CUG<sup>2</sup>&nbsp&nbsp&nbsp&nbsp;
HKUST<sup>3</sup>&nbsp&nbsp&nbsp&nbsp;
</div>

<div>
    <h4 align="center">
        <a href="https://arxiv.org/abs/2511.09388" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2511.09388-b31b1b.svg">
        </a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/cseeyangchen/Flora">
    </h4>
</div>
</div>

---

<h4>
This repo is the official implementation for "Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition." 
</h4>

![](src/framework.png)

---

## 📢 News
- **Feb 21, 2026:** Our paper is accepted by CVPR 2026 Findings Track!
- **Nov 14, 2025:** Our paper is now available on arXiv!
- **Nov 07, 2025:** This repository has been created, and the code has been uploaded!


## 📋 Contents
- [Data Preparation](#-data_preparation)
    - [A. From scratch](#-scratch)
    - [B. Pre-extracted features](#-features)
- [Compile CUDA Extensions for Shift-GCN](#-cuda)
- [Training](#-training)

## 🗂️ Data Preparation
<a id="-data_preparation"></a>

We provide two options for data preparation:
 - **From scratch**: You can download the raw skeleton sequences and extract the skeleton features yourself. We provide all the pre-trained Shift-GCN weights required for this process. Additionally, you can also train the Shift-GCN by yourself, following the procedure of Shift-GCN. Meanwhile, we also provide the training scripts, which are introduced below.
 - **Using pre-extracted features**: Alternatively, you can directly download the pre-extracted Shift-GCN skeleton features. We unify the feature formats from SynSE, SA-DAVE, STAR, and ours to ensure consistency across datasets. 

Further details about dataset splits can be found in our supplementary materials.

### A. From scratch
<a id="-scratch"></a>

#### A.1 Download Raw Datasets

```bash
# For NTU RGB+D 60 and 120
1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

# For PKU-MMD
1. Request and download the dataset here: https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html
2. Unzip all skeleton files from `Skeleton.7z` to `./data/pkummd_raw/part1`
3. Unzip all label files from `Label_PKUMMD.7z` to `./data/pkummd_raw/part1`
3. Unzip all skeleton files from `Skeleton_v2.7z` to `./data/pkummd_raw/part2`
4. Unzip all label files from `Label_PKUMMD_v2.7z` to `./data/pkummd_raw/part2`
```



#### A.2 Data Processing

Put downloaded data into the following directory structure:

```bash
data
  ├──ntu60  
  ├──ntu120   
  ├──nturgbd_raw   
  │   ├── nturgb+d_skeletons   
  │   │    └── .....   # from `nturgbd_skeletons_s001_to_s017.zip`
  │   └── nturgb+d_skeletons120  
  │        └── .....   # from `nturgbd_skeletons_s018_to_s032.zip`
  ├──pkummd
  └──pkummd_raw   
      ├── part1     
      │    └── .....   # from `Skeleton_v1.7z` and `Label_PKUMMD_v1.7z`
      └── part2
           └── .....   # from `Skeleton_v2.7z` and `Label_PKUMMD_v2.7z`
```

Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```bash
cd ./data/ntu60 # or cd ./data/ntu120

# Get skeleton of each performer
python get_raw_skes_data.py

# Remove the bad skeleton 
python get_raw_denoised_data.py

# Transform the skeleton to the center of the first frame
python seq_transformation.py
```

Generate PKU MMD I or PKU MMD II dataset:
```bash
cd ./data/pkummd/part1 # or cd ./data/pkummd/part2
mkdir skeleton_pku_v1 or mkdir skeleton_pku_v2

# Get skeleton of each performer
python pku_part1_skeleton.py or python pku_part2_skeleton.py

# Transform the skeleton to the center of the first frame
python pku_part1_gendata.py or python pku_part2_gendata.py

# Downsample the frame to 64
python preprocess_pku.py

# Concatenate train data and val data into one file
python pku_concat.py
```

#### A.3 Pretrain Skeleton Encoder (Shift-GCN) for Seen Classes 

If you would like to train [Shift-GCN](https://github.com/kchengiva/Shift-GCN) from scratch, please follow the procedure below. The best-performing pre-trained weights are stored in the `./pretrain_skeleton/save_models` directory.

```bash
# For NTU RGB+D 60 dataset (55/5 split):
cd pretrain_skeleton
python main.py --config config/ntu60/xsub_seen55_unseen5.yaml

# For NTU RGB+D 120 dataset (110/10 split):
cd pretrain_skeleton
python main.py --config config/ntu120/xsub_seen110_unseen10.yaml

# For PKU-MMD I dataset (46/5 split):
cd pretrain_skeleton
python main.py --config config/pku51/xsub_seen46_unseen5.yaml
```

For your convenience, we also provide the **Pre-trained Shift-GCN Weights** (STAR, STAR-SMIE, and PURLS Benchmark -- 1s-Shift-GCN). You can download them from [Google Drive](https://drive.google.com/file/d/16Q8Z-UmfCT0D7Ss4utrOyXFYjGo-10O7/view?usp=sharing), and place them in the `./pretrain_skeleton/save_models` directory.


### B. Pre-extracted features
<a id="-features"></a>

We also provide the **Pre-extracted Shift-GCN Skeleton Features** to unify the feature formats from SynSE, SA-DAVE, STAR, and Flora to ensure consistency across datasets. You can download them from [Google Drive](https://drive.google.com/file/d/1wrxetP_ItfJfxAfNPLdCRC2FZ8Mb9Pa1/view?usp=sharing), and place them in the following directory structure:

```bash
skeleton_features
  ├──synse_features   # 4s-Shift-GCN
  │   ├── ntu60_seen55_unseen5.npz
  │   ├── ntu60_seen48_unseen12.npz
  │   ├── ntu120_seen110_unseen10.npz
  │   └── ntu120_seen96_unseen24.npz
  │
  ├──sadave_features   # ST-GCN
  │   ├── ntu60_seen55_unseen5_split1.npz
  │   ├── ntu60_seen55_unseen5_split2.npz
  │   ├── ntu60_seen55_unseen5_split3.npz
  │   ├── ntu120_seen110_unseen10_split1.npz
  │   ├── ntu120_seen110_unseen10_split2.npz
  │   ├── ntu120_seen110_unseen10_split3.npz
  │   ├── pku51_seen46_unseen5_split1.npz
  │   ├── pku51_seen46_unseen5_split2.npz
  │   └── pku51_seen46_unseen5_split3.npz
  │
  ├──star_features   # 1s-Shift-GCN
  │   ├── ntu60_xsub_seen55_unseen5.npz
  │   ├── ntu60_xview_seen55_unseen5.npz
  │   ├── ntu60_xsub_seen48_unseen12.npz
  │   ├── ntu60_xview_seen48_unseen12.npz
  │   ├── ntu120_xsub_seen110_unseen10.npz
  │   ├── ntu120_xset_seen110_unseen10.npz
  │   ├── ntu120_xsub_seen96_unseen24.npz
  │   ├── ntu120_xset_seen96_unseen24.npz  
  │   ├── pku51_xsub_seen46_unseen5.npz
  │   ├── pku51_xview_seen46_unseen5.npz
  │   ├── pku51_xsub_seen39_unseen12.npz
  │   └── pku51_xview_seen39_unseen12.npz
  │
  ├──starsmie_features   # 1s-Shift-GCN
  │   ├── ntu60_xsub_seen55_unseen5_spli1.npz
  │   ├── ntu60_xsub_seen55_unseen5_spli2.npz
  │   ├── ntu60_xsub_seen55_unseen5_spli3.npz
  │   ├── pku51_xsub_seen46_unseen5_split1.npz
  │   ├── pku51_xsub_seen46_unseen5_split2.npz
  │   └── pku51_xsub_seen46_unseen5_split3.npz
  │
  └──flora_features   # 1s-Shift-GCN  
      ├── ntu60_xsub_seen40_unseen20.npz
      ├── ntu60_xsub_seen30_unseen30.npz
      ├── ntu120_xsub_seen80_unseen40.npz
      └── ntu120_xsub_seen60_unseen60.npz
```


## Compile CUDA Extensions for Shift-GCN
<a id="-cuda"></a>
```bash
cd models/shiftgcn/Temporal_shift
bash run.sh
```



## Training 
<a id="-training"></a>

```bash
# Train Flora on Basic Split Benchmark I (SynSE benchmark -- 4s-Shift-GCN) for the NTU-60 (55/5 Split)
python main.py --config configs/synse/ntu60_xsub_unseen5.yaml

# Train Flora on Basic Split Benchmark II (STAR benchmark -- 1s-Shift-GCN) for the NTU-60 (55/5 Split)
python main.py --config configs/star/ntu60_xview_unseen5_split1.yaml

# Train Flora on Random Split Benchmark I (SA-DAVE benchmark -- ST-GCN) for the NTU-60 (55/5 Split)
python main.py --config configs/sadave_random_split/ntu60_xsub_unseen5_split1.yaml

# Train Flora on Random Split Benchmark II (STAR & SMIE benchmark -- 1s-Shift-GCN) for the NTU-60 (55/5 Split)
python main.py --config configs/starsmie_random_split/ntu60_xsub_unseen5_split1.yaml

# Train Flora on More Challenging Seen-Unseen Benchmark (PURLS benchmark & our pre-trained features -- 1s-Shift-GCN) for the NTU-60 (40/20 Split)
python main.py --config configs/purls_flora/ntu60_xsub_unseen20.yaml

# Train Flora on Low-shot Training Sample （10% training data） Benchmark (SynSE benchmark -- 4s-Shift-GCN) for the NTU-60 (55/5 Split) 
python main.py --config configs/synse/ntu60_xsub_unseen5.yaml --low-shot --percentage 0.1
```
> **Note:** The default evaluation setting in the configuration file is `ZSL`. To evaluate under the GZSL setting, please change the `setting` in the configuration file to `GZSL`. 

## Acknowledgements
This repo is based on [Shift-GCN](https://github.com/kchengiva/Shift-GCN), [CrossFlow](https://github.com/qihao067/CrossFlow), [STAR](https://github.com/cseeyangchen/STAR), [Neuron](https://github.com/cseeyangchen/Neuron), and [TDSM](https://github.com/KAIST-VICLab/TDSM). Part of the pre-trained skeleton features is derived from [SynSE](https://github.com/skelemoa/synse-zsl) and [SA-DAVE](https://github.com/pha123661/SA-DVAE). The semantics for the NTU-series datasets are obtained from [SA-DAVE](https://github.com/pha123661/SA-DVAE)

Thanks to the authors for their work!

## Citation

Please cite this work if you find it useful:
```
@article{chen2025flora,
  title={Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition},
  author={Chen, Yang and Li, Miaoge and Rao, Zhijie and Zeng, Deze and Guo, Song and Guo, Jingcai},
  journal={arXiv preprint arXiv:2511.09388},
  url={https://arxiv.org/abs/2511.09388},
  year={2025}
}
```
