# FRAN: Unsupervised Cross-domain Fault Diagnosis Using Feature Representation Alignment Networks for Rotating Machinery
This is the Python+PyTorch code to reproduce the results of Fault Severity Diagnosis in paper ['Unsupervised Cross-domain Fault Diagnosis Using Feature Representation Alignment Networks for Rotating Machinery'](https://ieeexplore.ieee.org/document/9301443).

# Requirements
* Platform : Linux 
* Computing Environment:
  * CUDA 10.1 
  * TensorFlow 1.6.0
* Packages: ```pandas, numpy, scipy, argparse, tqdm```.
* Hardware (optional) : Nvidia GPU (requires around 7GB of GPU memory)

# Getting Started
1. Computing environment set up can be refered to [this repo](https://github.com/JiahongChen/Set-up-deep-learning-frameworks-with-GPU-on-Google-Cloud-Platform). 
1. Extract preprocessed [CWRU](https://csegroups.case.edu/bearingdatacenter/pages/welcome-case-western-reserve-university-bearing-data-center-website) data files in './CWRU_dataset'.
1. Run the code:
* For training:
```
bash batchrun.sh
```
* For visualization:
```
python correlationMatrix.py
```



# Citation
Please cite our paper and the dataset if you found them usefull.
```
@ARTICLE{chen2020unsupervised,
  author={J. {Chen} and J. {Wang} and J. {Zhu} and T. H. {Lee} and C. {De Silva}},
  journal={IEEE/ASME Transactions on Mechatronics}, 
  title={Unsupervised Cross-domain Fault Diagnosis Using Feature Representation Alignment Networks for Rotating Machinery}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMECH.2020.3046277}}
```
