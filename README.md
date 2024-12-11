## 1. Data Preparation
The directory structure of the whole project is as follows:
```
.
├── dataset
│   ├── Database134
│   │   ├── train
│   │   │   ├── images
│   │   │   └── manual
│   │   └── val
│   │       ├── images
│   │       └── manual
│   ├── DatasetCHUAC
│   │   └── train
│   │       ├── images
│   │       └── manual
│   ├── Custom
│   │      └──
├── model
│   └──
├── utils
│   └── 
└── main.py
```
## 2. Training
This is not very convenient for running custom configurations, which requires modifying the main.py code itself. We will refactor and optimize this in the near future.
- Run the train script.
```
python train.py
```
