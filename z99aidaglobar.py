
import copy
import datetime
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as utils_data
from scipy import stats
#from autoattack import AutoAttack

# Directories

DATAPATH  = 'D:\\Droploc\\Datadir'
CNETPATH  = 'D:\\Droploc\\CodeAidadefn\\Source_03\\Cnets_\\'
ROOTPATH  = 'D:\\Dropbox\\CodeAidadefn\\Source_03\\'
CONSFILE  = ROOTPATH + 'console.c'
BOARDFN0  = ROOTPATH + 'cznordo.c'
BOARDFN2  = ROOTPATH + 'cznordo2.c'
TNETFILE  = ROOTPATH + 'lastnet.pth'
DEBUGFNA =  ROOTPATH + 'Outdirb\\debuga'
DEBUGFNS =  ROOTPATH + 'Outdirb\\debugs'
WHTSFILE  = ROOTPATH + 'Outdirb\\dweights.c'
WHT1FILE  = ROOTPATH + 'Outdirb\\dweight1.c'

# General parameters

NLAYERS   = 6
BUTCHSZA  = 200
BUTCHEND  = (60000/BUTCHSZA)-1
BLRATE    = 0.0005
SBOL      = 0.0
BUDL      = 10.0
EPOBEG    = 3
EPOCHS    = 301
SUPCOE    = 1.000
SLOSSLB   = 0.050
UNSEPOCS  = 20
SATBSIZE  = 600
FORCECPU  = True

# Lr parameters

G2COE     = [0, 0.70, 0.70, 0.70, 0.70, 0.70]
LRTYP     = [0, 2.00, 2.00, 1.00, 1.00, 1.00]
XBIAS     = [0,-5.00,-5.00,-5.00,-5.00,-5.00]
UBIAS     = [0, 0.50, 1.00, 2.00, 2.00, 2.00]
LBIAS     = [0,-0.50,-1.00,-2.00,-2.00,-2.00]
TUPOR     = [0, 0.30, 0.30, 0.30, 0.30, 0.30]
FSLOP     = [0,  4.0,  4.0,  4.0,  4.0,  4.0]
FSLOQ     = [0, 40.0, 40.0, 40.0, 40.0, 40.0]

closed = ''' '''
LEFT      = 0
EAST      = 1
