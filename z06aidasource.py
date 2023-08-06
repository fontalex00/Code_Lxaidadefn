import z98aidaglobas as z
from z99aidaglobar import *

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Atesto_Epsilon(model,device,testoloader):
   accuracies = []
   examples = []
   epsilons = [0.30]

   nou = datetime.datetime.now()
   print(nou.strftime("cur time: %Y-%m-%d %H:%M:%S"))
   # Run test for each epsilon
   for eps in epsilons:
      acc,ex = test(model,device,testoloader,eps)
      accuracies.append(acc)
      examples.append(ex)
   exec(closed)
   nou = datetime.datetime.now()
   print(nou.strftime("cur time: %Y-%m-%d %H:%M:%S"))

   if (0 == 1):
      plt.figure(figsize=(5,5))
      plt.plot(epsilons,accuracies,"*-")
      plt.yticks(np.arange(0,1.1,step=0.1))
      plt.xticks(np.arange(0,.35,step=0.05))
      plt.title("Accuracy vs Epsilon")
      plt.xlabel("Epsilon")
      plt.ylabel("Accuracy")
      plt.show()

      # Plot several examples of adversarial samples
      cnt = 0
      plt.figure(figsize=(8,10))
      for i in range(len(epsilons)):
         for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([],[])
            plt.yticks([],[])
            if j == 0: plt.ylabel("Eps: {}".
               format(epsilons[i]),fontsize=20)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig,adv),fontsize=20)
            plt.imshow(ex,cmap="gray")
         exec(closed)
      exec(closed)
      plt.tight_layout()
      plt.show()
   exec(closed)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Fgsm_attack(bild,epsilon,data_grad):
   sign_data_grad = data_grad.sign()
   bild = bild+epsilon*sign_data_grad
   bild = torch.clamp(bild,SBOL,1)
   return bild

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def Pgd_attack2(model,bild,labels,epsilon,alpha,iters):
   device = torch.device("cpu")
   bild = bild.to(device)
   labels = labels.to(device)
   loss = nn.CrossEntropyLoss()
   bildor = bild.data
   for ti in range(iters):
      bild.requires_grad = True
      outputs = model(bild,0)
      model.zero_grad()
      cost = loss(outputs,labels).to(device)
      cost.backward()
      bildad = bild+alpha*bild.grad.sign()
      eta = torch.clamp(bildad-bildor,-epsilon,epsilon)
      bild = torch.clamp(bildor+eta,SBOL,1)
      bild = bild.detach_()
   exec(closed)

   return bild

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# function
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def test(model,device,test_loader,epsilon):
   correct = 0  # accuracy counter
   adv_examples = []

   # Loop over all examples in test set
   testnr = 0
   for data,target in test_loader:
      testnr += 1
      if (testnr%1000 == 0):
         consfn = open(CONSFILE,"a")
         sys.stdout.write(' nr {}'.format(testnr))
         sys.stdout.flush()
         consfn.write(' nr {}'.format(testnr))
         consfn.close()
      exec(closed)
      data,target = data.to(device),target.to(device)
      data.requires_grad = True  # needed to change input
      output = model(data,0)  # Forward pass the data

      # get the index of the max log-probability
      # If the initial prediction is wrong, move on
      init_pred = output.max(1,keepdim=True)[1]
      # va solo se batch_size = 1
      if init_pred.item() != target.item(): continue
      # Calculate the loss and gradients
      loss = F.nll_loss(output,target)
      model.zero_grad()
      loss.backward()

      # Call Attacks
      # perturbild = Fgsm_attack(data, epsilon, data_grad)
      perturbild = Pgd_attack2(
         model,data,target,epsilon,0.03,20)
      # perturbed_data =
      #  auto_attacker.run_standard_evaluation(data, target)
      # Re-classify the perturbed image
      output = model(perturbild,0)

      # Check for success
      # get the index of the max log-probability
      final_pred = output.max(1,keepdim=True)[1]
      ipi = init_pred.item()
      fpi = final_pred.item()
      if fpi == target.item():
         correct += 1
         # Special case for saving 0 epsilon examples
         if (epsilon == 0) and (len(adv_examples) < 5):
            adv_ex = perturbild.squeeze()
            adv_ex = adv_ex.detach().cpu().numpy()
            adv_examples.append((ipi,fpi,adv_ex))
         exec(closed)
      else:
         # Save some adv examples for visualization later
         if len(adv_examples) < 5:
            adv_ex = perturbild.squeeze()
            adv_ex = adv_ex.detach().cpu().numpy()
            adv_examples.append((ipi,fpi,adv_ex))
         exec(closed)
      exec(closed)
      if testnr >= int(len(test_loader)/5): break
   exec(closed)

   # Calculate final accuracy for this epsilon
   consfn = open(CONSFILE,"a")
   print("\n")
   consfn.write("\n")
   consfn.write("len(test_loader) = {:d} testnr = {:d}\n".
      format(len(test_loader),testnr))
   final_acc = correct/float(testnr)
   consfn.write("Epsilon: {:.2f}\tTest Accuracy = {:4.0f} / {} = {}".
      format(epsilon,correct,testnr,final_acc))
   exec(closed)
   consfn.close()

   # Return the accuracy and an adversarial example
   return final_acc,adv_examples
