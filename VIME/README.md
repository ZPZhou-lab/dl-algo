# VIME: Value Imputation and Mask Estimation

## Introduction
* **One Sentence Summary**: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain
* **Paper**: [VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain](https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf)
* **Keywords**: Self-supervised, Semi-supervised, Tabular Domain, AutoEncoder

## Quick Start
Prepare the python environment and install related dependencies.

```bash
# create python3.6 virtual environment
conda create -n vime python=3.6

# git clone
git clone https://github.com/ZPZhou-lab/dl-algo.git

# activate python environment
conda activate vime

# enter the directory and install the required packages
cd dl-algo/VIME/ && pip install -r requirements.txt
```

Test whether the model can run successfully through demo.

```bash
# enter the demo directory 
cd dl-algo/VIME/demo/

# run demo scripts
python titanic_self_demo.py
```