from z01aidasource import *
from z02aidasource import *
from z99aidaglobar import *
import z98aidaglobas as z
import z01aidasource as z01

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Module "Back-propagation"
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Copygrad(net,de,lr):
   b = net.nl[lr].bias
   u = net.nl[lr].weight
   if lr < NLAYERS-1: n = net.nx[lr]
   if lr == NLAYERS-1: n = net.nx[1] # to avoid error
   if de == 0: z.grold0[lr] = n.grad.detach().clone()
   if de == 1: z.ugrad1[lr] = u.grad.detach().clone()
   if de == 2: z.ugrad2[lr] = u.grad.detach().clone()
   if de == 1: z.bgrad1[lr] = b.grad.detach().clone()
   if de == 2: z.bgrad2[lr] = b.grad.detach().clone()

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Fixgrad(xgrad):
   xgrad[torch.isnan(xgrad)] = 0
   xgrad = torch.clamp(xgrad,-1.0,1.0)
   return xgrad

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Backprops(net,sot,output,labels):
   # Backprop, classific loss
   z.optimizer.zero_grad()
   lossx = z.criterion(output,labels)
   for lr in range(1,NLAYERS): net.nx[lr].retain_grad()
   lossx.backward(retain_graph=True)
   for lr in range(1,NLAYERS): Copygrad(net,0,lr)
   for lr in range(1,NLAYERS): Gradconc(lr,net,1)
   for lr in range(1,NLAYERS): Copygrad(net,1,lr)

   # Backprop, saturation loss
   at = torch.Tensor([0]).to(z.device)
   loss2 = z.criterion2(at,at)
   if z01.epocsatu and z.butch == 0:
      # grold factor
      #t0 = torch.zeros(10).to(z.device)
      #t0 = torch.unsqueeze(t0,0)
      #targetloss0 = t0.expand_as(output)
      #z.optimizer.zero_grad()
      #loss0 = z.criterion2(abs(output),targetloss0)
      #loss0.backward(retain_graph=True)
      #for lr in range(1,NLAYERS): Copygrad(net,0,lr)
      # saturation loss
      targetloss2 = torch.Tensor([0]).to(z.device)
      satloss,satloyr = Boardcalc(net,sot)
      for lr in range(1,NLAYERS):
         z.optimizer.zero_grad()
         # satloyt = satloyr[lr]
         satloyt = satloyr[lr].view(1)
         loss2 = z.criterion2(satloyt,targetloss2)
         loss2.backward(retain_graph=True)
         Gradconc(lr,net,2)
         Copygrad(net,2,lr)
      exec(closed)
      loss2 = z.criterion2(satloss,targetloss2)
   exec(closed)
   return lossx, loss2

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Boardcalc(net,sot):
   satloss = torch.zeros(1)
   satloyr = torch.zeros(NLAYERS)
   satloss = satloss.to(z.device)
   satloyr = satloyr.to(z.device)
   tscoe = 0
   for lr in range(1,NLAYERS):
      at = net.ns[lr]
      bt = sot.ns[lr]
      sze0 = list(at.size())[0]
      board = torch.ones(32,z.lrsize[lr])
      board = board.to(z.device)
      Boardsatu(at,bt,net,lr,board)
      Boardsat2(at,bt,net,lr,board)
      z.ford[lr][:32,:z.lrsize[lr]] += board[:32,:]
      sortal = torch.sort(board[1],descending=True)
      board[:] = board[:,sortal[1]]
      nu = int(sze0*0.0)
      # loss calc
      b1 = torch.sort(board[1],dim=0,descending=False)[0]
      rt = b1*Ranker(z.lrsize[lr],z.lrsize[lr],1)
      kt = 0.34*board[1]*board[0]+0.33*rt
      if lr <= 4: ct = 0.33*board[9]+kt
      if lr == 5: ct = 1.00*board[1]
      ct = torch.ones_like(ct)-ct
      cr = (ct)[nu:sze0].mean()
      lrcoe = 1
      # if lr == 1: lrcoe = 1
      tscoe += lrcoe
      satloss = satloss+cr*lrcoe
      satloyr[lr] = cr
   exec(closed)
   satloss = (satloss/tscoe)
   return satloss,satloyr

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Printcov(debugfn,score,bt,xt,lr):
   debugfn = open(debugfn,"w")
   for ri in range(100):
      debugfn.write("{:3d})".format(ri))
      debugfn.write("{:5.2f}".format(score[ri]))
      for ci in range(12):
         farg = int(bt[ri][ci]*100)
         if bt[ri][ci] >= 0.0:
            debugfn.write("{:4d}".format(farg))
         else: debugfn.write("    ")
      exec(closed)
      debugfn.write(" ___")
      for ci in range(12):
         farg = xt[ri][ci]
         if bt[ri][ci] >= 0.0:
            debugfn.write("{:4d}".format(farg))
         else: debugfn.write("    ")
      exec(closed)
      debugfn.write("\n")
   exec(closed)
   debugfn.close()

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Histosingle(xx):
   cbins = [-1.0]
   for ii in range(1,41): cbins.append(-1.0+ii*0.05)
   cn = np.histogram(xx,bins=cbins)[0]
   plt.hist(cbins[:-1],cbins,weights=cn)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Funcspect(xx,tshold,wsl,esl):
   at = torch.sigmoid(wsl*(xx-tshold))*1.1-0.1
   bt = torch.sigmoid(esl*(xx-tshold))*1.1-0.1
   return torch.where(xx-tshold < 0,at,bt)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Ranker(tsize,cover,expo):
   rf = torch.Tensor(tsize).to(z.device)
   rf.fill_(0.001)
   for ii in range(cover): rf[ii] = (1-(1/cover)*ii)**expo
   return rf *(tsize/rf.sum())

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Boardsatu(oldar,oldas,net,lr,board):
   znt = -XBIAS[lr]/BUDL
   for ii in range(1,-1,-1):
      if ii%2 == 0: bc = oldar
      if ii%2 == 1: bc = oldas
      bc = torch.transpose(bc,0,1)
      bc = torch.reshape(bc,(z.lrsize[lr],-1))
      xx = bc[0].clone().detach()
      if lr == 9: Histosingle(xx)
      rndel = 0.001*torch.rand(bc.size())
      rndel = rndel.to(z.device)
      f2 = 1*bc+bc*rndel # to avoid nan
      if ii%2 == 1:
         f2mean = torch.mean(f2,1,keepdim=True)
         f2std = torch.std(f2,1,keepdim=True)
         xx = torch.mean(f2)
         # f2 = torch.where(abs(f2)>0.01,f2,f2mean)
         xx = torch.mean(f2)
         tw = 0.10+f2std*0.5
         te = 0.10+f2std*0.5
         tw = torch.clamp(tw,0.10,0.30)
         te = torch.clamp(te,0.10,0.30)
      exec(closed)
      if ii%2 == 0: f3 = Funkast(bc,lr,0,znt,LEFT,tw,te)
      if ii%2 == 0: f5 = Funkast(bc,lr,0,znt,EAST,tw,te)
      if ii%2 == 1: f3 = Funkast(bc,lr,1,znt,LEFT,tw,te)
      if ii%2 == 1: f5 = Funkast(bc,lr,1,znt,EAST,tw,te)

      an = int(bc.size(dim=1)*0.20)
      wn = int(bc.size(dim=1)*0.20)
      en = int(bc.size(dim=1)*0.02)
      f4 = torch.where(f3 > f5,f3,f5)
      fs = torch.sort(f4,dim=1,descending=False)[0] # sorted
      fl = torch.topk(f4,an,dim=1,largest=False)[0] # lowest
      wt = torch.topk(f3,wn,dim=1,largest=True) [0] # largest west
      et = torch.topk(f5,en,dim=1,largest=True) [0] # largest east
      rt = fs*Ranker(bc.size(1),bc.size(1),2)

      board[2+ii*8] = torch.mean(f2,1)
      board[3+ii*8] = torch.mean(f3,1)
      board[4+ii*8] = torch.mean(f4,1)*0.0
      board[4+ii*8]+= torch.mean(rt,1)*1.0
      board[5+ii*8] = torch.mean(f5,1)
      board[6+ii*8] = torch.mean(wt,1)
      board[7+ii*8] = torch.mean(et,1)
   exec(closed)
   for no in range(z.lrsize[lr]): board[8][no] = no

   kn = 6+lr*0
   r3 = board[3].clone()
   r5 = board[5].clone()
   r4 = board[4].clone()
   r7 = board[7].clone()
   ds = torch.tanh(kn*(1-board[13]))
   bk = (r3-r5)/(0.01+abs((r3-r5)))
   st = (bk+1)*0.5*r7 # to backward pass error
   #st = torch.where(st < 0,st,torch.tanh(2*st))
   vc = (0.0+1.0*r4) # void centre
   if lr <= 4: board[1] = st*vc
   if lr == 5: board[1] = st*vc
   board[1] = torch.tanh(2*board[1])
   # if the west and east piles are different, moving a
   # point to the saturated pile won’t determine any gain

   at = z.grold0[lr]
   at = torch.transpose(at,0,1)
   at = torch.reshape(at,(z.lrsize[lr],-1))
   #at = torch.where(at > 0,at,0)
   an = int(at.size(dim=1)*1.00)
   rt = Ranker(at.size(1),at.size(1),2) #xx =rt.sum()
   rt = torch.unsqueeze(rt,0).expand_as(at)
   #bt = torch.topk(at,an,dim=1,largest=True)[0]
   bt = torch.sort(at,dim=1,descending=True)[0]
   bt = torch.mean(bt*rt,1)
   ct = torch.sum(bt)
   board[0] = (bt/ct)*z.lrsize[lr]
   xx = board[0].sum()
   if lr == NLAYERS-1: board[0] = 1
   if z.epoch < UNSEPOCS: board[0] = 1

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Boardsat2(oldar,oldas,net,lr,board):
   znt = -XBIAS[lr]/BUDL
   at = net.nx[lr].clone()
   #at = Funkast(at,lr,0,znt,EAST)
   if LRTYP[lr] == 2:
      atshape = (SATBSIZE,z.lrsize[lr],-1)
      at = torch.reshape(at,atshape)
      ftshape = (SATBSIZE,z.lrsize[lr],10)
      ft = torch.zeros(ftshape).to(z.device)
      for ii in range(0,SATBSIZE):
         idx = torch.randperm(at.size(2))[:10]
         ft[ii,:,:] = at[ii,:,idx]
      exec(closed)
      at = torch.max(ft,2)[0]
   exec(closed)
   mean0 = torch.mean(at,0,keepdim=True)
   mean1 = torch.mean(at,1,keepdim=True)
   #at = at-mean0.expand_as(at)
   at = nn.functional.leaky_relu(at)

   b0 = board[0].clone()
   xx = b0.sum()
   b1 = board[1].clone()
   cover = min([20,z.lrsize[lr]])
   rt = Ranker(z.lrsize[lr],cover,1)
   xx = rt.sum()
   b0 = torch.unsqueeze(b0,0).expand_as(at)
   b1 = torch.unsqueeze(b1,0).expand_as(at)
   rt = torch.unsqueeze(rt,0).expand_as(at)

   sortvalx = 0.3*at*b0+0.7*at
   xt = torch.sort(sortvalx,dim=1,descending=True)[1]
   at = torch.gather(at,1,xt)
   b0 = torch.gather(b0,1,xt)
   b1 = torch.gather(b1,1,xt)
   et = at*b1*rt
   ft = torch.mean(et,1)   
   if z01.epocstat and lr <= 3:
      fnamea = DEBUGFNA + str(lr) + '.c'
      fnames = DEBUGFNS + str(lr) + '.c'
      Printcov(fnamea,ft,at,xt,lr)
      Printcov(fnames,ft,b1,xt,lr)
   exec(closed)
   #bt = torch.transpose(at,0,1)
   #rndel = 0.001*torch.rand(bt.size())
   #bt = 1*bt+bt*rndel # to avoid nan
   #board[9] = 0.50*(1-torch.mean(torch.corrcoef(bt)))
   board[9] = torch.mean(ft)
   board[torch.isnan(board)] = 0

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Gradconc(lr,net,opt):
   lrshape = (z.lrsize[lr],-1)
   #if lr <= 2 and opt == 1: return
   u = net.nl[lr].weight
   absuht = abs(u).detach()
   absgrd = abs(u.grad).detach()
   at = absuht.view(lrshape).max(dim=1)[0]
   bt = absgrd.view(lrshape).max(dim=1)[0]
   at[at == 0] = 1
   bt[bt == 0] = 1
   #at = torch.where(at != 0,at,torch.ones_like(at))
   #bt = torch.where(bt != 0,bt,torch.ones_like(bt))
   absuht /= Unsquandu(at,absuht,lr)
   absgrd /= Unsquandu(bt,absgrd,lr)

   wco = net.uc[lr].clone()
   if lr <= 1: wct = 0.58
   if lr >= 2: wct = 0.38
   wco = torch.sigmoid(99*(wco-wct))
   wco = Unsquandu(wco,absuht,lr)
   wco = wco.to(z.device)
   if lr <= 1: expcoe = 0.85-wco*0.20
   if lr >= 2: expcoe = 0.85-wco*0.20

   u.grad *= absuht**expcoe
   u.grad *= absgrd**(0.50)
   at = abs(u.grad).view(lrshape)
   ht = torch.max(at,1)[0]
   ct = ht/torch.max(at)
   ht = ht/(ct**0.0)
   ht = Unsquandu(ht,u.grad,lr)
   u.grad *= 1/ht  # in [0,1] interval
   at = abs(u.grad).view(lrshape)
   xx = torch.max(at,1)[0]
   yy = torch.max(at,1)[1]
   u.grad = Fixgrad(u.grad)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Gradadjust(net,loss2):
   # ugrad1 adjustement
   supcoe = SUPCOE
   if z.epoch < UNSEPOCS: supcoe = 0
   for lr in range(1,NLAYERS):
      coe = abs(z.ugrad1[lr]).max()
      if coe == 0: coe = 1
      z.ugrad1[lr] *= (1.0*supcoe/coe)  # 0.010
      z.bgrad1[lr] *= (0.5*supcoe/coe)  # 0.005
      z.ugrad1[lr] = Fixgrad(z.ugrad1[lr])
      z.bgrad1[lr] = Fixgrad(z.bgrad1[lr])
      z.gdmean[lr] = abs(z.ugrad1[lr]).mean()
   exec(closed)

   # ugrad2 adjustement
   for lr in range(1,NLAYERS):
      ar = abs(z.ugrad1[lr]).max()
      br = abs(z.ugrad2[lr]).max()
      if ar == 0: ar = 0.1
      if br == 0: br = 1.0
      cr = (ar/br)*G2COE[lr]
      if br*cr <= 0.001: cr = 0.001/br
      if loss2 <= SLOSSLB: cr = 0
      z.ugrad2[lr] *= cr  # correction wht
      z.bgrad2[lr] *= cr  # correction bias
      z.ugrad2[lr] = Fixgrad(z.ugrad2[lr])
      z.bgrad2[lr] = Fixgrad(z.bgrad2[lr])
      z.gxmean[lr] = abs(z.ugrad2[lr]).mean()
   exec(closed)

   # Gradient mixer
   for lr in range(1,NLAYERS):
      b = net.nl[lr].bias
      u = net.nl[lr].weight
      u.grad = z.ugrad1[lr]+z.ugrad2[lr]
      b.grad = z.bgrad1[lr]+z.bgrad2[lr]
      u.grad = torch.clamp(u.grad,-1,1)
   exec(closed)
