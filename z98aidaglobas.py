from z99aidaglobar import *
import numpy as np
import torch

# Globals read /write
device = "cpu"
epoch = 0
butch = 0
dlossx = 0
dloss2 = 0
daccur = 0
butchsze = 0
criterion = 0
criterion2 = 0
optimizer = 0
lrsize = np.zeros(NLAYERS,dtype=int)
insel = np.zeros((NLAYERS,9))
ugrad1 = np.zeros(NLAYERS,dtype=np.object)
ugrad2 = np.zeros(NLAYERS,dtype=np.object)
bgrad1 = np.zeros(NLAYERS,dtype=np.object)
bgrad2 = np.zeros(NLAYERS,dtype=np.object)
grold0 = np.zeros(NLAYERS,dtype=np.object)
gdmean = np.zeros(NLAYERS)
gxmean = np.zeros(NLAYERS)
ford = torch.zeros((NLAYERS,32,512))
ubias = [0,0,0,0,0]
content = ""
