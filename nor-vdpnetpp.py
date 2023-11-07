#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import argparse

import numpy as np

from qmodel import QModel

#
#
#
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='NoR-VDPNet++', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run', type=str, help='Application [tonemapping|inversetonemapping|hdr_compression|sdr_distortion')
    parser.add_argument('data', type=str, help='Base dir of run to evaluate')

    args = parser.parse_args()

    if args.run == 'tonemapping':
        checkpoint = 'weights/weight_tmo'

    if args.run == 'inversetonemapping':
        checkpoint = 'weights/weight_itmo'

    if args.run == 'hdr_compression':
        checkpoint = 'weights/weight_hdrc'

    if args.run == 'sdr_distortion':
        checkpoint = 'weights/weight_sdr'
        
    model = QModel(checkpoint, 2, 1400, (args.grayscale == 1))
    
    #run for each file
    names_hdr = [f for f in os.listdir(args.data) if f.endswith('.hdr')]
    names_exr = [f for f in os.listdir(args.data) if f.endswith('.exr')]

    names_jpg = [f for f in os.listdir(args.data) if f.endswith('.jpg')]
    names_png = [f for f in os.listdir(args.data) if f.endswith('.png')]

    names = names_hdr + names_exr + names_jpg + names_png
    
    for name in names:
        fn = os.path.join(args.data, name)

        p_model = model.predict(fn)

        print(name + " Q: " + str(round(p_model * 10000)/100))
