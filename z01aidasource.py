from z01aidasource import *
from z02aidasource import *
from z03aidasource import *
from z04aidasource import *
from z06aidasource import *
import z98aidaglobas as z
from z99aidaglobar import *

#pdb.set_float_format('%g')
global epocbase
global epocsatu
global epocstat
global trainloader
global checkloader
global testoloader
global net,sit,sot
global startzeit
global startepoc

if __name__ == "__main__": Main()
def Main():
   global epocbase
   global epocsatu
   global epocstat
   global trainloader
   global checkloader
   global testoloader
   global net,sit,sot
   global startzeit
   global startepoc

   Initstuff()
   Initstuff2()
   if 1 == 1: Schaufunc(net)
   startzeit = 0
   startepoc = 0
   for z.epoch in range(epocbase,EPOCHS):
      for z.butch,data in enumerate(trainloader,0):
         z.butchsze = BUTCHSZA
         if z.butch%20 == 0:
            astr = ' epoch {} butch {}'
            print(astr.format(z.epoch,z.butch))
         exec(closed)

         # conditions
         epocinit = (z.epoch-epocbase <= EPOBEG)
         epocstat = epocinit or z.epoch%10 == 0
         epocsatu = epocinit or z.epoch%2 == 0
         if epocsatu and z.butch == 0:
            data = next(iter(checkloader))
            z.butchsze = SATBSIZE
         exec(closed)

         # inits vars
         if epocstat and z.butch == 0:
            z.dlossx = 0  # loss classification
            z.dloss2 = 0  # loss saturation
            z.daccur = 0  # accuracy measure
            z.insel = np.zeros_like(z.insel)
            z.ford = torch.zeros_like(z.ford)
         exec(closed)
         # calc concentration
         for lr in range(1,NLAYERS):
            u = net.nl[lr].weight
            net.uc[lr] = Whtconcallo(u,lr)
            net.uc[lr] = net.uc[lr].to(z.device)
         exec(closed)

         # Forward propagation
         Uhtnormal(net)
         inputs,labels = data
         inputs = inputs.to(z.device)
         labels = labels.to(z.device)
         inputs.requires_grad = True
         output = net(inputs,0)
         lastlr = net.nx[5]
         softlr = torch.nn.Softmax(output)
         # Forward propagation, scrambled
         if epocsatu and z.butch == 0:
            Auxilo_Scrogen(net,sot,sit)
            imagesn = net.nx[0].view(z.butchsze,1,28,28)[:20]
            imagess = sit.nx[0].view(z.butchsze,1,28,28)[:20]
            bt = torch.cat((imagesn,imagess),0)
            # Bildschau(torchvision.utils.make_grid(bt))
         exec(closed)

         # Back propagation and step
         lossx, loss2_ = Backprops(net,sot,output,labels)
         if epocsatu and z.butch == 0: loss2 = loss2_
         Gradadjust(net,loss2)
         u.grad[torch.isnan(u.grad)] = 0
         z.optimizer.step()
         z.dlossx += lossx.detach()
         z.dloss2 += loss2.detach()
         output = output.float()
         bestind = torch.max(output,1).indices
         bestval = torch.max(output,1).values
         z.daccur += torch.mean((bestind == labels).float())

         if epocstat and z.butch == 0:
            DsamplInsel(net)
            Histoplotter(net,sot,0)
            Histoplotter(net,sot,1)
            Histoplotter(net,sot,2)
            PrintoUhtval(net,sot)
            PrintoUhtshp(net,sot)
         exec(closed)
         if epocstat and z.butch == BUTCHEND:
            PrintoBoard(0)
            #PrintoBoard(2)
            # PruneUeights(net)
         exec(closed)
         if epocstat and z.butch == BUTCHEND or \
            (z.epoch == epocbase and z.butch == 0):
            Printo_Consoleb(lastlr,output)
            torch.save(net,TNETFILE)
         exec(closed)
         if z.epoch%20 == 0 and z.butch == BUTCHEND:
            filepath = CNETPATH+'trainednet_'
            filepath += str(z.epoch)+'.pth'
            torch.save(net,filepath)
         exec(closed)

      exec(closed)
      gc.collect()  # end epoch
   print('Finished Training');
   net.eval()

   if 1 == 1:
      Atesto_Epsilon(net,"cpu",testoloader)
   exec(closed)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Initstuff():
   global epocbase
   global trainloader
   global checkloader
   global testoloader

   print(sys.version)
   root = '../data'
   print(root)
   print(" CUDA Available: ",torch.cuda.is_available())
   if (torch.cuda.is_available()):
      z.device = torch.device("cuda")
   exec(closed)
   if FORCECPU == True: z.device = torch.device("cpu")
   print(" z.device value: ",z.device)
   epocbase = 0
   if os.path.isfile(CONSFILE):
      # os.remove(CONSFILE)
      consfn = open(CONSFILE,"r")
      inhalt = consfn.read()[:18]
      epocbase = [int(s) for s
         in inhalt.split() if s.isdigit()][0]
      consfn.close()
   exec(closed)
   torch.autograd.set_detect_anomaly(True)
   # Test neu functions
   # Test neu functions_End

   if SBOL == -1.0: ar = 0.50; br = 0.50
   if SBOL == 0.0: ar = 0.00; br = 1.00
   if SBOL == 0.2: ar = -0.25; br = 1.25
   trans = transforms.Compose([
      # transforms.ToTensor()])
      transforms.ToTensor(),
      transforms.Normalize((ar,),(br,))])
   an = BUTCHSZA
   bn = SATBSIZE
   trainset = datasets.MNIST(root=DATAPATH,train=True,
      download=True,transform=trans)
   testoset = datasets.MNIST(root=DATAPATH,train=False,
      download=True,transform=trans)
   trainloader = utils_data.DataLoader(trainset,an,True,
      num_workers=1)
   checkloader = utils_data.DataLoader(trainset,bn,True,
      num_workers=1)
   testoloader = utils_data.DataLoader(testoset, 1,True,
      num_workers=1)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Initstuff2():
   global checkloader
   global net,sit,sot

   inputs = next(iter(checkloader))[0]
   net = Netb(inputs)
   print(" ",net)
   sit = Netb(inputs)
   sot = Netb(inputs)
   for lr in range(NLAYERS):
      sit.nx[lr] = sit.nx[lr].detach()
      sot.nx[lr] = sot.nx[lr].detach()
   exec(closed)

   random.seed(0)
   np.random.seed(0)
   torch.manual_seed(0)
   if os.path.isfile(TNETFILE):
      net = torch.load(TNETFILE)
   exec(closed)
   # criterion = nn.NLLLoss()
   z.criterion = nn.CrossEntropyLoss()
   z.criterion2 = nn.L1Loss()
   # criterion2= nn.MSELoss()
   z.optimizer = optim.SGD(net.parameters(),BLRATE,0.9)

   Model2device(net,z.device)
   Model2device(sot,z.device)
   Model2device(sit,z.device)
   xx = net.nx[lr].is_cuda
   z.ford = z.ford.to(z.device)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Schaufunc(net):
   plt.rcParams['font.sans-serif'] = "Arial"
   #plt.rcParams['font.family'] = 'monospace'
   for lr in range(1,6):
      x = np.linspace(-1.0,1.0,100)
      xt = torch.Tensor([x])
      foutb = torch.squeeze(xt)
      #foutb = torch.squeeze(torch.sigmoid(8*(xt-0.5)))
      #foutb = 1-foutb*(1-foutb)*4
      znt = -XBIAS[lr]/BUDL
      ts = TUPOR[lr]
      #ts = ts.expand_as(x)
      fout0 = torch.squeeze(Funkout(xt,lr))
      func3 = torch.squeeze(Funkast(xt,lr,0,znt,LEFT,ts,ts))
      func5 = torch.squeeze(Funkast(xt,lr,0,znt,EAST,ts,ts))
      funcs = torch.squeeze(Funkast(xt,lr,1,znt,LEFT,0.1,0.1))
      func4 = torch.where(func3 > func5,func3,func5)
      func6 = torch.squeeze(Funcspect(xt,0.5,4,18))
      fig = plt.figure()
      ax = fig.add_subplot(111)
      plt.axhline(0,color='black')
      plt.axvline(0,color='black')
      plt.rc('xtick',labelsize=15)
      plt.rc('ytick',labelsize=15)
      plt.xticks(np.arange(-1.0,1.1,0.5))
      plt.yticks(np.arange(-1.0,1.1,0.2))
      ax.yaxis.tick_left()
      plt.tight_layout(pad=0)
      plt.subplots_adjust(left=0.1,right=0.9)
      ax.plot(x,foutb.numpy(),'#000000',linewidth=3)
      ax.plot(x,fout0.numpy(),'#444444',linewidth=3)
      ax.plot(x,func3.numpy(),'#8b0000',linewidth=3)
      ax.plot(x,funcs.numpy(),'#008b00',linewidth=3)
      ax.plot(x,func5.numpy(),'#00008b',linewidth=3)
      ax.plot(x,func6.numpy(),'#8b008b',linewidth=3)
      ax.annotate("lr="+str(abs(lr)),xy=(0.8,-0.8),
         weight='bold',size=14)
      # ax.set_aspect(aspect=0.80) # 1.30 for figures
      fig.set_size_inches(8,5)
      fig.canvas.manager.window.move(40,70)
      plt.show(block=False)
      plt.pause(2);
      plt.close()
   exec(closed)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Model2device(model,device):
   model = model.to(device)
   for lr in range(0,NLAYERS):
      if lr >= 0: model.nx[lr] = model.nx[lr].to(device)
      if lr >= 1: model.ns[lr] = model.ns[lr].to(device)
   exec(closed)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Printo_Consoleb(lastlr,output):
   global startzeit
   global startepoc

   deltazeit = (time.perf_counter()-startzeit)
   deltaepoc = (z.epoch-startepoc)*60
   if deltaepoc != 0: deltazeit /= deltaepoc
   startzeit = time.perf_counter()
   startepoc = z.epoch

   z.content = ""
   if os.path.isfile(CONSFILE):
      consfn = open(CONSFILE,"r")
      z.content = consfn.read()
      z.content = z.content[:20000]  # crops
      consfn.close()
      os.remove(CONSFILE)
   exec(closed)
   consfn = open(CONSFILE,"w")
   an = z.butch+1
   consfn.write("\n Ep:{:4d} {:d}".format(z.epoch,an))
   consfn.write(" dloss: {:5.3f}".format(z.dlossx/an))
   consfn.write(" dloss2:{:6.3f}".format(z.dloss2/an))
   consfn.write(" daccur:{:6.3f}".format(z.daccur/an))
   consfn.write("\nlast lr: ")
   for ci in range(10):
      consfn.write("{:6.2f}".format(lastlr[0][ci]))
   exec(closed)
   consfn.write("\n output: ")
   for ci in range(10):
      consfn.write("{:6.2f}".format(output[0][ci]))
   exec(closed)

   # insel
   for pi in range(-2,10):
      if pi == -2: consfn.write("\n gdmean: ")
      if pi == -1: consfn.write("\n gxmean: ")
      if pi == 1: consfn.write("\n puconc: ")
      if pi == 2: consfn.write("\n  whtsp: ")
      if pi == 3: consfn.write("\n  whtsn: ")
      if pi == 4: consfn.write("\n   bias: ")
      if pi == 5: consfn.write("\n  hones: ")
      if pi == 6: consfn.write("\n  hvals: ")
      if pi == 7: consfn.write("\n  nord1: ")
      if pi == 8: consfn.write("\nalnord1: ")
      if pi == 9: consfn.write("\nalnord9: ")
      for lr in range(1,NLAYERS):
         an = z.insel[lr][0]
         ford1 = z.ford[lr][1][:z.lrsize[lr]]
         b1 = torch.sort(ford1,descending=False)[0]
         rankar = Ranker(z.lrsize[lr],z.lrsize[lr],1)
         rt = b1*rankar
         ct = z.ford[lr][0]*z.ford[lr][1]
         dt = z.ford[lr][1]
         et = z.ford[lr][9]
         if pi > -3: astr = " ({:7.5f})"
         if pi >= 1: astr = " ({:7.2f})"
         if pi >= 4: astr = " ({:7.3f})"
         if pi ==-2: printedvar = z.gdmean[lr]
         if pi ==-1: printedvar = z.gxmean[lr]
         if pi >= 1 and pi <= 6: printedvar = z.insel[lr][pi]/an
         if pi == 7: printedvar = ct[:z.lrsize[lr]].mean()
         if pi == 8: printedvar = dt[:z.lrsize[lr]].mean()
         if pi == 9: printedvar = et[:z.lrsize[lr]].mean()
         if pi != 0: consfn.write(astr.format(printedvar))
      exec(closed)
   exec(closed)
   # duration
   bstr = " durat (min): {:4.1f} ___\n\n\n"
   consfn.write(bstr.format(deltazeit))
   consfn.write(z.content)
   consfn.close()

