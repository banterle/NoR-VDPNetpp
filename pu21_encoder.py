#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import numpy as np

class PU21Encoder:
    
    def __init__(self, type = 'banding_glare'):
        self.L_min = 0.005;
        self.L_max = 10000;
        
        if type == 'banding':
            self.par = [1.070275272, 0.4088273932, 0.153224308, 0.2520326168, 1.063512885, 1.14115047, 521.4527484]
        elif type == 'banding_glare':
            self.par = [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142]
        elif 'peaks':
            self.par = [1.043882782, 0.6459495343, 0.3194584211, 0.374025247, 1.114783422, 1.095360363, 384.9217577]
        else: #'peaks_glare':
            self.par = [816.885024, 1479.463946, 0.001253215609, 0.9329636822, 0.06746643971, 1.573435413, 419.6006374]
        
    #
    #
    #
    def apply(self, L):
        epsilon = 1e-5;
        np.clip(L, self.L_min, self.L_max)
        p = self.par;
        tmp = (p[0] + p[1] * np.power(L, p[3])) / (1.0 + p[2] * np.power(L, p[3]))
        V = p[6] * (np.power(tmp, p[4]) - p[5])
        return V


#
#
#
def PU21Encode(x):
    pu21 = PU21Encoder()
    return pu21.apply(x)