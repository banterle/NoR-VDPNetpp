#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys
import argparse
from qmodel import QModel
from util import read_img_cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval Q regressor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, help='HDR_COMP (JPEG-XT compression), HDR_ITMO (inverse tone mapping), SDR (distortions for 8-bit images), and and SDR_TMO (tone mapping distortions).')
    parser.add_argument('img_folder', type=str, help='Base dir of run to evaluate')
    parser.add_argument('-dr', '--displayreferred', type=str, default='yes', help='Do we need to apply the display? (yes/no)')
    parser.add_argument('-cs', '--colorspace', type=str, default='REC709', help='Color space of the input images')
    parser.add_argument('--color', type=str, default='gray', help='Enable/Disable color inputs')

    args = parser.parse_args()
    
    bGrayscale = (args.color == 'gray')
        
    if args.mode == 'SDR':
        try:
            model = QModel('weights/norvdpnetpp_sdr.pth', btype = 2, grayscale = True, colorspace = args.colorspace, display_referred = args.displayreferred)
        except:
            model = QModel('http://www.banterle.com/francesco/work/norvdpnetpp/norvdpnetpp_sdr.pth', btype = 2, grayscale = True, colorspace = args.colorspace, display_referred = args.displayreferred)

    elif args.mode == 'HDR_COMP':
        try:
            model = QModel('weights/norvdpnetpp_hdrc.pth', btype = 2, grayscale = True, colorspace = args.colorspace, display_referred = args.displayreferred)
        except:
            model = QModel('http://www.banterle.com/francesco/work/norvdpnetpp/norvdpnetpp_hdrc.pth', btype = 2, grayscale = True, colorspace = args.colorspace, display_referred = args.displayreferred)
                    
    elif args.mode == 'HDR_ITMO':
        try:
            model = QModel('weights/norvdpnetpp_itmo.pth', btype = 2, grayscale = True, colorspace = args.colorspace, display_referred = args.displayreferred)
        except:
            model = QModel('http://www.banterle.com/francesco/work/norvdpnetpp/norvdpnetpp_itmo.pth', btype = 2, grayscale = True, colorspace = args.colorspace, display_referred = args.displayreferred)

    elif args.mode == 'SDR_TMO':
        try:
            model = QModel('weights/norvdpnetpp_tmo.pth', btype = 2, grayscale = True, colorspace = args.colorspace, display_referred = args.displayreferred)
        except:
            model = QModel('http://www.banterle.com/francesco/work/norvdpnetpp/norvdpnetpp_tmo.pth', btype = 2, grayscale = True, colorspace = args.colorspace, display_referred = args.displayreferred)

    else:
        print('The mode ' + args.mode + ' selected is not supported.')
        print('Supported modes: HDR_ITMO, HDR_COMP, SDR, and SDR_TMO.')
        sys.exit()

    names_mat = [f for f in os.listdir(args.img_folder) if f.endswith('.mat')]
    names_hdr = [f for f in os.listdir(args.img_folder) if f.endswith('.hdr')]
    names_exr = [f for f in os.listdir(args.img_folder) if f.endswith('.exr')]
    names_hdr = sorted(names_mat + names_hdr + names_exr)

    names_jpg = [f for f in os.listdir(args.img_folder) if f.lower().endswith('.jpg')]
    names_jpeg = [f for f in os.listdir(args.img_folder) if f.lower().endswith('.jpeg')]
    names_png = [f for f in os.listdir(args.img_folder) if f.lower().endswith('.png')]
    names_sdr = sorted(names_jpg + names_jpeg + names_png)
    
    names = names_hdr + names_sdr
    
    for name in names:
        fn = os.path.join(args.img_folder, name)
        p_model = float(model.predict(fn))
        print(name + " Q: " + str(round(p_model * 10000)/100))

    del model
