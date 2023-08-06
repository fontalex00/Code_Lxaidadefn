import z02aidasource as z02
import z98aidaglobas as z
from z99aidaglobar import *

def Bildschau(bild):
   bild = bild/2+0.5  # unnormalize
   npbild = bild.cpu().numpy()
   plt.imshow(np.transpose(npbild,(1,2,0)))
   plt.show()

def Bsinvoid(x):
   x[x <= 0.01] = 0.01
   x[x >= 0.99] = 0.99
   z = -torch.log(1/x-1)*0.1+0.5
   return z

def Topbot(at,an,lopt):
   return torch.topk(at,an,largest=lopt)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function _ KRITISCH
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def DsamplInsel(net):
   nlrs = NLAYERS  # abbreviation
   #insar = np.zeros(8)
   for lr in range(1,nlrs):
      insar = np.zeros(8)
      uht = net.nl[lr].weight.detach()
      xuht,sind,pt = z02.Xsumuht(uht,lr,0)
      xuht,sind,nt = z02.Xsumuht(uht,lr,1)
      uconc = net.uc[lr]
      losco = z.ford[lr][0][:z.lrsize[lr]] # losco = 1
      losco = losco.detach().clone()/z.lrsize[lr]
      bias = XBIAS[lr] + net.nl[lr].bias
      insar[1] += torch.dot(losco,uconc)
      insar[2] += torch.dot(losco,abs(pt))
      insar[3] += torch.dot(losco,abs(nt))
      insar[4] += torch.dot(losco,bias)
      for bi in range(10):
         bt = net.nx[lr][bi].detach().clone()
         if lr <= 2: bt = torch.mean(bt,(1,2))
         bt = torch.sigmoid(20*(bt-0.5))
         insar[5] += torch.dot(losco,bt)/10
      exec(closed)
      for ii in range(1,7):
         z.insel[lr,ii] += insar[ii]
      z.insel[lr,0] += 1
   exec(closed)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Histoplotter(net,sot,opt):
   nlrs = NLAYERS  # abbreviation
   for lr in range(nlrs-1,0,-1):
      for rs in range(2):
         if opt == 0:
            if rs == 0: at = net.ns[lr].detach().clone()
            if rs == 1: at = sot.ns[lr].detach().clone()
            at = torch.transpose(at,0,1)
            at = torch.reshape(at,(z.lrsize[lr],-1))
         exec(closed)
         if opt == 1:
            if rs == 0: at = z.ugrad1[lr]
            if rs == 1: at = z.ugrad2[lr]
            xx = abs(z.ugrad1[lr]).max()
         exec(closed)
         if opt == 2:
            at = net.nl[lr].weight.detach().clone()
         exec(closed)
         len1 = list(at.size())[1]
         if opt != 2: cbins = [-1.0]
         if opt == 2: cbins = [-5.0]
         for ii in range(1,41):
            if opt != 2: cbins.append(-1.0+ii*0.05)
            if opt == 2: cbins.append(-5.0+ii*0.25)
         exec(closed)
         bt = z.ford[lr][0][:z.lrsize[lr]]
         #nart = torch.sort(-bt*(1-0))[1]
         nart = torch.sort(bt,descending=True)[1]
         ti = 0  # option
         an = min(10,z.lrsize[lr])
         for ho in range(an):
            if ti == 0: no = nart[ho]
            if ti == 1: no = nart[z.lrsize[lr]-1-ho]
            if no < z.lrsize[lr]:
               cn = np.histogram(at[no].cpu(),bins=cbins)[0]
               if no == 109 or no == 190: exec(closed)
               if opt == 0: astr = 'Outdirbout/0'+str(lr)
               if opt == 1: astr = 'Outdirgrad/0'+str(lr)
               if opt == 2: astr = 'Outdirxwht/0'+str(lr)
               if rs == 0: bstr = astr+'besto_histor'
               if rs == 0: wstr = astr+'worst_histor'
               if rs == 1: bstr = astr+'besto_histos'
               if rs == 1: wstr = astr+'worst_histos'
               hostr = str(ho)
               if ho < 10: hostr = '0'+ hostr
               bstr = bstr+hostr+'.png'
               wstr = wstr+hostr+'.png'
               if rs == 0: col = '#1E90FF'
               if rs == 1: col = '#FF6347'
               plt.hist(cbins[:-1],cbins,weights=cn,color = col)
               #density=True)
               txt = "nd nr="+str(no.item())
               plt.annotate(txt,xy=(-0.8,0),weight='bold',size=14)
               plt.ylim(0,len1/5)
               if ti == 0: plt.savefig(bstr)
               if ti == 1: plt.savefig(wstr)
               plt.clf()
            exec(closed)
         exec(closed)
         plt.close()
      exec(closed)
   exec(closed)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def PrintoUhtval(net,sot):
   nlrs = NLAYERS  # abbreviation
   fn = open(WHTSFILE,"w+")
   fn.write("\n Layer: {:3d}\n".format(0))
   #for no in range(z.lrsize[0]):
   #   if no%100 == 0: fn.write("\n ner ")
   #   fn.write("({:3d}){:6.3f} ".format(no,net.nx[0][0][no]))
   #exec(closed)

   fn.write("\n")
   for lr in range(1,nlrs):
      bias = XBIAS[lr]+net.nl[lr].bias
      u = net.nl[lr].weight
      if lr <= 2: u = u.view(z.lrsize[lr],-1)
      fn.write("\n Layer: {:3d}".format(lr))
      bt = z.ford[lr][0]  # 1
      bt = bt[:z.lrsize[lr]]
      nart = torch.sort(-bt*(1-0))[1]
      for ho in range(10):
         no = nart[ho]
         totp = u.data[no][u.data[no] > 0].sum()
         totn = u.data[no][u.data[no] < 0].sum()
         fn.write("\n\n nd  ({:3d})".format(no))
         fn.write(" bias{:6.2f}".format(bias[no]))
         fn.write(" totp{:6.2f}".format(totp))
         fn.write(" totn{:6.2f}\n".format(totn))
         htar = np.argsort(-abs(u.data[no].cpu()))
         # ztar= np.sort(-abs(whts[no]))
         for hi in range(20):
            if hi%5 == 0: fn.write("\n wht ")
            ni = htar[hi]
            astr = "({:3d}){:6.3f} "
            fn.write(astr.format(ni,u.data[no][ni]))
         exec(closed)
      fn.write("\n")
   fn.close()

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def PrintoPixels(fn,lr,whts,lp,rp):
   # pixel shapes _
   if LRTYP[lr] == 1:
      for yi in range(1,28):
         for xi in range(28):
            pxl = whts[lp][yi*28+xi]
            fn.write("{:10.7f} ".format(pxl))
         exec(closed)
         fn.write(" _ _ ")
         for xi in range(28):
            pxl = whts[rp][yi*28+xi]
            fn.write("{:10.7f} ".format(pxl))
         exec(closed)
         fn.write("\n")
      exec(closed)
   exec(closed)
   if LRTYP[lr] == 2:
      fn.write(11*"\n")
      for yi in range(0,5):
         fn.write(121*" ")
         for xi in range(0,5):
            pxl = whts[lp][0][xi][yi]
            fn.write("{:10.7f} ".format(pxl))
         exec(closed)
         fn.write(132*" ")
         fn.write(" _ _ ")
         fn.write(121*" ")
         for xi in range(0,5):
            pxl = whts[rp][0][xi][yi]
            fn.write("{:10.7f} ".format(pxl))
         exec(closed)
         fn.write("\n")
      exec(closed)
      fn.write(11*"\n")
   exec(closed)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def PrintoUhtshp(net,sot):
   lr = 1  # lr 1 only
   at = z.ford[1][1]
   bt = z.ford[1][0]
   lart = torch.sort(-bt*(1-0))[1]
   rart = torch.sort(-at*(1-0))[1]
   for nd in range(512):
      if lart[nd] >= z.lrsize[1]: lart[nd] = 0   
      if rart[nd] >= z.lrsize[1]: rart[nd] = 0   
   exec(closed)
   if len(rart) == 0: rart = [0]
   fn = open(WHT1FILE,"w+")
   whts = net.nl[1].weight.data

   for ho in range(int(z.lrsize[1]/1)):
      h1 = min(ho,len(rart)-1)
      astr = "nd:{:3d}|{:3d}"+' '*79
      bstr = "xx:{:5.2f}"+' '*69
      bstr += "xx:{:5.2f}"+' '*69+"xx:{:5.2f}"
      fn.write("\n")
      for ti in range(3):
         for ni in range(2):
            if ni == 0: no = lart[ho]
            if ni == 1: no = rart[h1]
            if ti == 0: b1 = 1
            if ti == 0: b2 = z.ford[lr][0][no]
            if ti == 0: b3 = z.ford[lr][1][no]
            if ti == 1: b1 = z.ford[lr][2][no]
            if ti == 1: b2 = z.ford[lr][4][no]
            if ti == 1: b3 = z.ford[lr][5][no]
            if ti == 2: b1 = z.ford[lr][10][no]
            if ti == 2: b2 = z.ford[lr][12][no]
            if ti == 2: b3 = z.ford[lr][13][no]
            if ni == 0: fn.write(astr.format(ho,no))
            if ni == 1: fn.write(astr.format(h1,no))
            fn.write(bstr.format(b1,b2,b3))
            fn.write(" "*62)
         exec(closed)
         fn.write("\n")
      exec(closed)
      # Pixels _
      PrintoPixels(fn,lr,whts,lart[ho],rart[h1])
      fn.write("\n")
   exec(closed)
   fn.close()

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def PrintoBoard(rowstart):
   if rowstart == 0: boardfn = open(BOARDFN0,"w")
   if rowstart == 2: boardfn = open(BOARDFN2,"w")
   kord = z.ford.clone()
   boardfn.write("({:2d},{:3d})".format(-1,-1))
   for ii in range(0,8):
      boardfn.write(" {:4d}_".format(ii))
   boardfn.write("\n")
   for lr in range(1,NLAYERS):
      bt = z.ford[lr][0][:z.lrsize[lr]]
      nart = torch.sort(bt,descending=True)[1]
      for no in range(10):
         for rs in range(rowstart,rowstart+2):
            boardfn.write("({:2d},{:3d})".format(lr,no))
            for ii in range(0,8):
               ar = z.ford[lr][ii+rs*8][nart[no]]
               boardfn.write("{:6.2f}".format(ar))
            exec(closed)
            if rs == rowstart+0: boardfn.write("\n")
            if rs == rowstart+1: boardfn.write("\n\n")
         exec(closed)
      exec(closed)
   exec(closed)
   boardfn.close()
