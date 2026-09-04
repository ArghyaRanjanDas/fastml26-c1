# FastML 2026 Hackathon Challenges

> **Team fastml26-c1 results (Challenge 1):** see [`team/README.md`](team/README.md) — headline: 2.7k-param student, AUC 0.909 float / 0.906 synthesized on a VU9P, 253k LUT, 1,692 DSP, 0.42 µs, one SLR.


Resources and instructions for use during the FastML2026 hackathon. More details on the challenges and competition can be found on the [conference website.](https://indico.cern.ch/event/1654479/page/43940-hackathon)

---

## Overview + JupyterHub 

The [National Research Platform](https://nrp.ai/) and [San Diego Supercomputer Center](https://www.sdsc.edu/) are generously providing computing resources for the hackathon, namely a JupyterHub instance with dedicated GPU and FPGA resources, along with container images pre-built for ML and FPGA workflows. Participants are not required to make use of these resources, but it is highly recommended. 

All datasets for the challenges can be found **pre-loaded in the JupyterHub** in the `\hack-data\` directory, organized by challenge. All datasets are organized into `train` and `eval` subsets. Please use the `train` data to train and test your networks during the hackathon. The `eval` datasets are for presenting final results at the end of the hackathon. Do not use `eval` in training. 

If you filled out the hackathon signup form, you are already added to the JupyterHub. Please notify us if you did not sign up or have trouble logging in. 

**The JupyterHub is located here: [https://fastml26-hack.nrp-nautilus.io/](https://fastml26-hack.nrp-nautilus.io).**

Log in with your university/institution or CERN credentials if applicable.

You should see something like this screen when setting up your server after you log in:
<img width="627" height="803" alt="image" src="https://github.com/user-attachments/assets/7858dfaa-1a88-455b-9a25-cfef7bd2471b" />

We recommend using the **NRP Deep Learning & Data Science Full, PyTorch** or **NRP Deep Learning & Data Science Full, Tensorflow** images. 

Each participant should select the following resources per person:
* GPUs: 1
* Cores: 8
* RAM: 50

If your team wants to do a dedicated longer training **one person can use all of the resources of the whole team**. For example, for a team of 5, one person can request:
* GPUs: 1*5=5
* Cores: 8*5=40
* RAM: 50*5=250
* FPGAs: 0

Be sure to shut down everyone else's servers. Please do not request more than this, or your server will likely not start at all.

If you want to go crazy and actually try to deploy on an FPGA, use the FPGA image:
* GPUs: 0
* Cores: 4
* RAM: 50
* FPGA: 1

**Only 1 person per team can use a FPGA server at the same time**. Please do not train on the FPGA nodes, and do not add both GPUs and a FPGA to your server. **Do not request more than 1 FPGA**.

The region and GPU type should stay as "general". It doesn't matter. 

40 A10 GPUs and 7 Alveo U55C FPGAs are reserved for use during the hackathon. 

---

## Challenge Setup

Teams can choose between 4 challenges: 3 based on the [COLLIDE2V](https://huggingface.co/datasets/fastmachinelearning/collide-2v) High Energy Physics open dataset and 1 based on Meta's [emg2pose](https://github.com/facebookresearch/emg2pose) motion-capture dataset. 

The data per challenge can be found in the /hack-data/ directory.

```bash
#collide2v hh->4b challenge
hack-data/C1_HH4b/train/
hack-data/C1_HH4b/eval/

#collide2v foundation model challege
hack-data/C5_foundation_model/train/
hack-data/C5_foundation_model/eval/

#collide2v robust tagging challenge
hack-data/C9_robust_tagging/train/
hack-data/C9_robust_tagging/eval/

#emd2pose motion capture challenge
hack-data/emg2pose/emg2pose_dataset_full/train/
hack-data/emg2pose/emg2pose_dataset_full/eval/
#also a smaller dataset for the tutorial/testing it out
hack-data/emg2pose/emg2pose_dataset_mini/
```

**Introductory notebooks** are provided for both the COLLIDE2V ([collide2v_intro.ipynb](https://github.com/quinnanm/fastml26-hackathon/blob/main/collide2v_intro.ipynb)) and emg2pose ([emg2pose_intro.ipynb](https://github.com/quinnanm/fastml26-hackathon/blob/main/emg2pose_intro.ipynb)) challenges. Please start with those! Once your server starts you can open a terminal and clone this repo to get the intro notebooks:

```bash
git clone https://github.com/quinnanm/fastml26-hackathon.git
```

Most dependencies should be included in the images, but a few extras may be needed. Open a terminal, install them with `pip`, and **restart your notebooks after installing new dependencies**. Note you will need to install any dependencies again each time you start a new server. 

The following, for example, is needed for the intro notebooks (at least): 

```
# collide2v_intro
pip install awkward

# emg2pose_intro
pip install -e emg2pose/
pip install h5py==3.11.0 hydra-core==1.3.2 omegaconf joblib==1.4.2 tqdm
```

For FPGA/hls4ml/quantization resources we recommend getting started with the [hls4ml tutorials](https://github.com/fastmachinelearning/hls4ml-tutorial/tree/main), namely 1_getting_started and 2_quantization. **You can add this on to the PyTorch/Tensorflow server you are using.** The following (maybe more) is needed to use those tutorials:

```bash
#hls4ml tutorial intro notebooks
git clone https://github.com/fastmachinelearning/hls4ml-tutorial.git

conda env create -f environment.yml -p ~/envs/hls4ml-tutorial
source ~/.bashrc
conda activate ~/envs/hls4ml-tutorial
python -m ipykernel install --user --name hls4ml-tutorial --display-name "hls4ml-tutorial"

#Source the vitis/vivado license:
source /tools/Xilinx/Vivado/2023.1/settings64.sh

#if get XILINX_VITIS path error, add this to top of cell 2:
import os
os.environ['XILINX_VITIS'] = '/tools/Xilinx/Vitis/2024.2'
```



