import z04aidasource as z04
import z98aidaglobas as z
from z99aidaglobar import *

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Module "Forward propagation"
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Unsquandu(at,uht,lr):
   #bt = at.unsqueeze_(pos)
   bt = torch.unsqueeze(at,-1)
   if LRTYP[lr] == 2: bt = torch.unsqueeze(bt,-1)
   if LRTYP[lr] == 2: bt = torch.unsqueeze(bt,-1)
   bt = bt.expand_as(uht)
   return bt

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Funksin(x):
   sr = 6+10*((x-0.5)*2)**6
   y = torch.sigmoid(sr*(x-0.5))
   return x

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function _ KRITISCH
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Funkout(x,lr):
   tshold = -XBIAS[lr]/BUDL
   sr = 20+14*x**6
   s = torch.where(x < tshold,20,sr)
   y = torch.sigmoid(s*(x-tshold))
   d = (0.03*(x+1.1)+0.00)**1
   e = (0.06*(x-1.0)+1.00)
   y = torch.clamp(y,d,e)
   return y

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function _ KRITISCH
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Funkast(x,lr,rs,aor,side,tw,te):
   if side == 0: x = 2*aor-x
   if side == 1: x = 0*aor+x
   if side == 0: ts = tw
   if side == 1: ts = te
   threshold = aor
   d = 0.04*(x+1.1)**1
   f = torch.tanh(5*(x-threshold))
   f = torch.where(f <= d,d,f)
   f = torch.where(x-threshold >= ts,1,f)
   f = torch.clamp(f,0,1)*1.4-0.4
   return f

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function _ KRITISCH
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Funkous(x,lr):
   return x

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function _ KRITISCH
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Whtconcallo(u,lr):
   uhtar = u.data.clone()
   res = torch.ones(z.lrsize[lr])
   for no in range(z.lrsize[lr]):
      uht = uhtar[no]
      uht[uht < 0] = 0
      uht /= torch.sum(uht)
      xx = torch.sum(uht)
      if lr <= 2: uht = uht.view(-1)
      uhtcnc = torch.sum((uht/xx)**2)
      res[no] = uhtcnc**(1/2)
      res[no][torch.isnan(res[no])] = 0
   exec(closed)
   return res

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Btransposer(xt,lr):
   at = torch.transpose(xt,0,1)
   if LRTYP[lr] == 2: at = torch.transpose(at,1,2)
   if LRTYP[lr] == 2: at = torch.transpose(at,2,3)
   return at

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function _ KRITISCH
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Xsumuht(uhts,lr,ii):
   xuht = uhts.clone()
   sind = uhts*((-1)**ii)
   xuht[sind < 0] = 0
   if LRTYP[lr] == 1: setofd = 1
   if LRTYP[lr] == 2: setofd = (1,2,3)
   xsu = torch.sum(abs(xuht),setofd)
   return xuht,sind,xsu

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function _ KRITISCH
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Uhtnormal(net):
   for lr in range(1,NLAYERS):
      uhts = net.nl[lr].weight.data
      for ii in range(2):
         xuht,sind,xsu = Xsumuht(uhts,lr,ii)
         xsu[xsu == 0] = 1
         at = torch.Tensor([2*z.epoch/100]) # 900
         budl = BUDL*(2.0 - torch.tanh(at))
         budl = budl.to(z.device)
         if ii == 0: coe = abs(budl/xsu)
         if ii == 1: coe = 1.0*(xsl/xsu)
         coe = torch.clamp(coe,0,1)
         if ii == 0: xsl = xsu.clone()*coe
         coe = Unsquandu(coe,uhts,lr)
         xuht = xuht*coe
         uhts[sind > 0] = xuht[sind > 0]
      exec(closed)
      bias = net.nl[lr].bias.data
      bias[bias < LBIAS[lr]] = LBIAS[lr]
      bias[bias > UBIAS[lr]] = UBIAS[lr]
   exec(closed)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function _ KRITISCH
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Lrforwarc(net,ney,lr,xc):
   if lr <= 2:
      yc = net.nl[lr](xc)
      xc = Funkout(yc/BUDL,lr)
      sc = Funkous(yc/BUDL,lr)
      ney.nx[lr] = net.pool(xc)
      ney.ns[lr] = net.pool(sc)
      xc = ney.nx[lr]
   exec(closed)
   if lr >= 3:
      if lr == 3: xc = xc.view(-1,16*4*4)
      if lr == 5: xc = ney.nx[3]
      if lr == 5: xc = torch.cat((xc,ney.nx[4]),dim=1)
      yc = net.nl[lr](xc)
      ney.nx[lr] = Funkout(yc/BUDL,lr)
      ney.ns[lr] = Funkous(yc/BUDL,lr)
      xc = ney.nx[lr]
   exec(closed)
   return yc,xc

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# class
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Netb(nn.Module):
   def __init__(self,x):
      super(Netb,self).__init__()
      self.pool = nn.MaxPool2d(2,2)
      self.conv1 = nn.Conv2d(1,12,kernel_size=5)
      self.conv2 = nn.Conv2d(12,16,kernel_size=5)
      self.fc3 = nn.Linear(16*4*4,120)
      self.fc4 = nn.Linear(120,84)
      self.fc5 = nn.Linear(120+84,10)

      z.lrsize[0] = 1
      z.lrsize[1] = 12
      z.lrsize[2] = 16
      z.lrsize[3] = 120
      z.lrsize[4] = 84
      z.lrsize[5] = 10

      self.nx = np.empty(NLAYERS,dtype=np.object)
      self.ns = np.empty(NLAYERS,dtype=np.object)
      self.nl = np.empty(NLAYERS,dtype=np.object)
      self.uc = np.empty(NLAYERS,dtype=np.object)
      self.nl[1] = self.conv1
      self.nl[2] = self.conv2
      self.nl[3] = self.fc3
      self.nl[4] = self.fc4
      self.nl[5] = self.fc5
      self.forward(x,0)

   def forward(self,x,opt):
      x = Funksin(x)
      self.nx[0] = x
      xc = x
      for lr in range(1,NLAYERS):
         yc, xc = Lrforwarc(self,self,lr,xc)
      exec(closed)
      return yc

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Auxilo_Scrogen(net,sot,sit):
   # scrambled /creates sit
   for lr in range(0,NLAYERS):
      if lr in range(0,3):
         if lr == 0: an = 28
         if lr == 1: an = 12
         if lr == 2: an = 4
         ln = z.lrsize[lr]
         oldt = (z.butchsze,ln,an,an)
         neut = (z.butchsze,ln,an*an)
         nt = torch.reshape(net.nx[lr],neut)
         kt = torch.reshape(sit.nx[lr],neut)
         nt = torch.transpose(nt,1,2)
         kt = torch.transpose(kt,1,2)
         drawsize = int(an*an*0.30)
         for bi in range(0,z.butchsze):
            for ri in range(6):
               bn = random.randrange(z.butchsze)
               idx = torch.randperm(an*an)[:drawsize]
               kt[bi][idx] = nt[bn][idx].detach()
            exec(closed)
         exec(closed)
         kt = torch.transpose(kt,2,1)
         sit.nx[lr] = torch.reshape(kt,oldt)
      exec(closed)
      if lr in range(3,NLAYERS) and 1 == 1:
         ln = z.lrsize[lr]
         drawsize = int(ln*0.30)
         for bi in range(0,z.butchsze):
            for ri in range(6):
               bn = random.randrange(z.butchsze)
               idx = torch.randperm(int(ln))[:drawsize]
               sit.nx[lr][bi][idx] = net.nx[lr][bn][idx].detach()
            exec(closed)
         exec(closed)
      exec(closed)
   exec(closed)

   # scrambled /updates sot
   sot.nx[0] = sit.nx[0]
   for lr in range(1,NLAYERS):
      Lrforwarc(net,sot,lr,sit.nx[lr-1])
   exec(closed)

