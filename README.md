# FastML 2026 Hackathon Challenges

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

For training your networks/looking at the data, we recommend using the **NRP Deep Learning & Data Science Full, PyTorch** or **NRP Deep Learning & Data Science Full, Tensorflow** images. 

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

For hls4ml/FPGA workflows/resource estimates and model quantization, we recommend the **FPGA** image. When using the FPGA image you should request the following:
* GPUs: 0
* Cores: 4
* RAM: 50
* FPGA: 1

The region and GPU type should stay as "general". It doesn't matter. 

40 A10 GPUs and 7 Alveo U55C FPGAs are reserved for use during the hackathon. 

---

## Challenge Setup

Teams can choose between 4 challenges: 3 based on the [COLLIDE2V](https://huggingface.co/datasets/fastmachinelearning/collide-2v) High Energy Physics open dataset and 1 based on Meta's [emg2pose](https://github.com/facebookresearch/emg2pose) motion-capture dataset. 

The data per challenge can be found in the /hack-data/ directory.

```bash
/folders
```

Introductory notebooks are provided for both the COLLIDE2V (collide2v_intro.ipynb) and emg2pose (emg2pose_intro.ipynb) challenges. Please start with those! Once your server starts you can open a terminal and clone this repo to get the intro notebooks:

```bash
git clone
```

Most dependencies should be included in the images, but a few extras may be needed. Open a terminal, install them with `pip`, and **restart your notebooks after installing new dependencies**. Note you will need to install any dependencies again each time you start a new server. 

The following, for example, is needed for the intro notebooks:

```bash
```


For FPGA/hls4ml/quantization resources on FPGA image servers, we recommend getting started with the [hls4ml tutorials](https://github.com/fastmachinelearning/hls4ml-tutorial/tree/main), namely 1_getting_started and 2_quantization. From a FPGA server on the JupyterHub, the following is needed to use those tutorials:

```bash
```



