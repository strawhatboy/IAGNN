# Intention Adaptive Graph Neural Network (IAGNN)

This is the official repository of paper *Intention Adaptive Graph Neural Network for Category-Aware Session-Based Recommendation*.

![Model](docs/model_whitebg.png "IAGNN")

If you found this work helpful, please kindly cite the paper as follows:

```bibtex
@inproceedings{DBLP:conf/dasfaa/CuiSZPZGW22,
  author    = {Chuan Cui and
               Qi Shen and
               Shixuan Zhu and
               Yitong Pang and
               Yiming Zhang and
               Hanning Gao and
               Zhihua Wei},
  editor    = {Arnab Bhattacharya and
               Janice Lee and
               Mong Li and
               Divyakant Agrawal and
               P. Krishna Reddy and
               Mukesh K. Mohania and
               Anirban Mondal and
               Vikram Goyal and
               Rage Uday Kiran},
  title     = {Intention Adaptive Graph Neural Network for Category-Aware Session-Based
               Recommendation},
  booktitle = {Database Systems for Advanced Applications - 27th International Conference,
               {DASFAA} 2022, Virtual Event, April 11-14, 2022, Proceedings, Part
               {II}},
  series    = {Lecture Notes in Computer Science},
  volume    = {13246},
  pages     = {150--165},
  publisher = {Springer},
  year      = {2022},
  url       = {https://doi.org/10.1007/978-3-031-00126-0\_10},
  doi       = {10.1007/978-3-031-00126-0\_10},
  timestamp = {Fri, 29 Apr 2022 14:50:39 +0200},
  biburl    = {https://dblp.org/rec/conf/dasfaa/CuiSZPZGW22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Prerequisite

Install the dependencies by `conda`

```bash
dgl~=0.6.0.post1
ipdb~=0.13.9
numpy~=1.21.2
pretty_errors~=1.2.24
PyMySQL~=1.0.2
scikit_learn~=1.0.2
torch~=1.8.1
TorchSnooper~=0.8
tqdm~=4.62.3
```

or by `pip`:

```bash
pip install -r requirements.txt
```

## Dataset

[GoogleDrive](https://drive.google.com/drive/folders/1ZuR55uY50QPYNygo3Tn2zeYVmr4Ab_Ta?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1Chi5PxGX5NM-iM4oL-JrsQ) (提取码：2jd1)

Put the downloaded `*.pkl` files by following this file structure:

```bash
|--dataset
   |--diginetica_x
      |--train.pkl
      |--test.pkl
   |--jdata_cd
      |--train.pkl
      |--test.pkl
   |--yc_BT_4
      |--train.pkl
      |--test.pkl
|--IAGNN	# Souce code of this repository
   |--train.py
   |--IAGNN.py
   ...
```

## How to train

```bash
# JData
python train.py --lr=0.003 --lr_step=2 --GL=3 --dataset=jdata_cd
# Yoochoose
python train.py --lr=0.001 --lr_step=1 --GL=1 --dataset=yc_BT_4
# Diginetica
python train.py --lr=0.003 --lr_step=1 --GL=2 --dataset=diginetica_x
```
