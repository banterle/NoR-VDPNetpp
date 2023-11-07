NoR-VDPNet++
============
NoR-VDPNet++ is a deep-learning based no-reference metric trained on [HDR-VDP](http://hdrvdp.sourceforge.net/wiki/).
Traditionally, HDR-VDP requires a reference image, which is not possible to have in some scenarios.

![HDR-VDP](images/hdrvdp.png?raw=true "HDR-VDP")

NoR-VDPNet++ is a no-reference metric, so it requires a single image in order to asses its quality. NoR-VDPNet can be trained on High Dynamic Range (HDR) images or Standard Dynamic Range (SDR) images (i.e., classic 8-bit images).

![NoR-VDPNet++](images/our.png?raw=true "NoR-VDPNet++")


DEPENDENCIES:
==============

Requires the PyTorch library along with Image, NumPy, SciPy, Matplotlib, glob2, pandas, and scikit-learn.

As the first step, you need to follow the [instructions for installing PyTorch](http://pytorch.org/).

To install dependencies, please use the following command: 

```bash
pip3 install numpy, scipy, matplotlib, glob2, pandas, image, scikit-learn, opencv-python. 
```

HOW TO RUN IT:
==============
To run our metric on a folder of images (i.e., JPEG, PNG, EXR, and HDR, files), you need to launch the file ```nor-vdpnetpp.py```; for example:

```
python3 nor-vdpnetpp.py tonemapping /home/user00/images
```

WEIGHTS DOWNLOAD:
=================
Coming soon.

DO NOT:
=======

1) Please do not use weights_sdr for HDR images;

2) Please do not use weights_hdrc for SDR images;

3) Please do not use weights_hdrc for testing distortions that are not JPEG-XT distortions or compression distortions;

4) Please do not use weights_sdr for distortions that are not in the paper.

DATASET PREPARATION:
====================
Coming soon.

TRAINING:
=========
Coming soon.


REFERENCE:
==========

If you use NoR-VDPNet in your work, please cite it using this reference:


@ARTICLE{10089442,

  author={Banterle, Francesco and Artusi, Alessandro and Moreo, Alejandro and Carrara, Fabio and Cignoni, Paolo},

  journal={IEEE Access}, 

  title={NoR-VDPNet++: Real-Time No-Reference Image Quality Metrics}, 

  year={2023},

  volume={11},

  number={},

  pages={34544-34553},

  doi={10.1109/ACCESS.2023.3263496}
}