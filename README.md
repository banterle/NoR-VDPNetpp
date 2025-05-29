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
To run our metric on a folder of images (i.e., JPEG, PNG, EXR, HDR, and MAT files),
you need to launch the file ```norvdpnet.py```. Some examples:

Testing SDR images for the trained distortions (see the paper):

```
python3 norvdpnetpp.py SDR /home/user00/images_to_be_sdr/
```

Testing HDR images after JPEG-XT compression:

```
python3 norvdpnetpp.py HDR_COMP /home/user00/images_to_be_hdr/
```

Testing HDR images after tone mapping operators:

```
python3 norvdpnetpp.py SDR_TMO /home/user00/images_to_be_sdr/
```

Testing images after inverse tone mapping operators:

```
python3 norvdpnetpp.py HDR_ITMO /home/user00/images_to_be_hdr/
```

WEIGHTS DOWNLOAD:
=================
Weights can be downloaded here:
 <a href="http://www.banterle.com/francesco/work/norvdpnetpp/norvdpnetpp_sdr.pth">SDR</a>, 
 <a href="http://www.banterle.com/francesco/work/norvdpnetpp/norvdpnetpp_hdrc.pth">HDRC</a>,
  <a href="http://www.banterle.com/francesco/work/norvdpnetpp/norvdpnetpp_tmo.pth">TMO</a>, and
  <a href="http://www.banterle.com/francesco/work/norvdpnetpp/norvdpnetpp_itmo.pth">ITMO</a>.

Note that these weights are meant to model ONLY determined distortions; please see reference to have a complete overview.


DO NOT:
=======

There are many people use NoR-VDPNet++ in an appropriate way:

1) Please do not use weights_nor_sdr for HDR images;

2) Please do not use weights_nor_jpg_xt for SDR images;

3) Please do not use weights_nor_tmo for HDR images; only gamma-encoded SDR images!!!

4) Please do not use weights_nor_itmo for SDR images;

5) Please do not use weights for different distortions.

DATASET PREPARATION:
====================
Coming soon.

TRAINING:
=========
Coming soon.


REFERENCE:
==========

If you use NoR-VDPNet in your work, please cite it using this reference:

```
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
```
