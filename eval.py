#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import argparse

import numpy as np

from qmodel import QModel


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Eval Q regressor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run', type=str, help='Base dir where weights are')
    parser.add_argument('data', type=str, help='Base dir of run to evaluate')
    parser.add_argument('--btype', type=int, default = 2, help='Base dir of run to evaluate')
    parser.add_argument('--maxClip', type=float, default = 1400, help='Maximum luminance output of the display - set at training time!')
    parser.add_argument('--grayscale', type=int, default=1, help='Grayscale')


    args = parser.parse_args()

    model = QModel(args.run, args.btype, args.maxClip, (args.grayscale == 1))
    
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

    del model
