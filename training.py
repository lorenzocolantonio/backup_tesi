import numpy as np
import math
import time
import torch
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Modules.training_functions import *
from Modules.pennylane_functions import *

# if gpu available, set device to gpu
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using the GPU")
else:
    device = torch.device("cpu")
    print("WARNING: Could not find GPU, using the CPU")
ld_dim=8
T=5
# load dataset
mnist_images0 = np.load(f'Data/dataset_ld_{ld_dim}_0.npy')
mnist_images1 = np.load(f'Data/dataset_ld_{ld_dim}_1.npy')

mnist_images =np.concatenate((mnist_images0, mnist_images1), axis = 0)
print(np.shape(mnist_images))

np.random.shuffle(mnist_images)
mnist_images = torch.tensor(mnist_images).to(device)

# make dataloader
data_loader = torch.utils.data.DataLoader(mnist_images, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
qc_array=np.array([0,2,4])
min_array=np.array([0.1,0.05,0.01,0.08])
layer_array=np.array([5,10,20,50])
print(NUM_QUBITS)
print(T)
for layer_indx in range(len(layer_array)):
    n_layer=layer_array[layer_indx]
    for q_indx in range(len(qc_array)):
        qc=qc_array[q_indx]
        for min_indx in range(len(min_array)):
            min_b=min_array[min_indx]

            betas      = np.insert(np.linspace(10e-8,min_b, T), 0, 0)
            print(np.shape(betas))
            alphas     = 1 - betas
            alphas_bar = np.cumprod(alphas)
            pi         = math.pi
            betas      = torch.tensor(betas).float().to(device)
            alphas     = torch.tensor(alphas).float().to(device)
            alphas_bar = torch.tensor(alphas_bar).float().to(device)
            theta_1    = Variable(torch.rand((n_layer*3*NUM_QUBITS+n_layer*3*(NUM_QUBITS)), device = device), requires_grad=True)
            optimizer = torch.optim.Adam([theta_1], lr = LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = SCHEDULER_PATIENCE, gamma = SCHEDULER_GAMMA, verbose = False)
            trained_thetas_1 = []
            loss_history = []
            best_loss = 1e10

            for epoch in range(NUM_EPOCHS):
                print(epoch)

                t0 = time.time()
                num_batch=0
                tot_loss=0

                for image_batch in data_loader:

                    # extract batch of random times and betas
                    t = torch.randint(0, T, size = (BATCH_SIZE, ), device=device)
                    betas_batch = betas[t].to(device)
                    alphas_batch=alphas_bar[t].to(device)

                    # assemble input at t add noise (t+1)
                    target_batch = assemble_input(image_batch, t, alphas_bar,ld_dim ,device)
                    input_batch  = noise_step(target_batch, t+1, betas,ld_dim, device)
                    target_batch = target_batch / torch.norm(target_batch, dim = 1).view(-1, 1)
                    input_batch  = input_batch / torch.norm(input_batch, dim = 1).view(-1, 1)
                    #zero = torch.zeros(128, 256*7).to(device)

                    # concatenate the two tensors along the second dimension
                    #input_batch = torch.cat((input_batch, zero), dim=1)
                    #target_batch = torch.cat((target_batch, zero), dim=1)
                    # Feed to circuit, compute the loss and update the weights
                    num_batch+=1
                    loss = loss_fn_aq(qc,theta_1,n_layer, input_batch, target_batch)
                    tot_loss+=loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # append parameters and print loss
                trained_thetas_1.append(theta_1.cpu().clone().detach().numpy())

                loss_history.append(tot_loss/num_batch)
                if loss.item()< best_loss:
                    best_loss=loss.item()

                # implement learning rate scheduler
                scheduler.step()


            # print every epoch
                print(f'T={T} Epoch: {epoch+1}/{NUM_EPOCHS} - Loss: {loss.item():.4f} b_loss={best_loss:.4f} - T: {time.time()-t0:.2f}s/epoch ,tempo_previto={((time.time()-t0)*(NUM_EPOCHS-1-epoch+NUM_EPOCHS*(len(qc_array)-q_indx-1)+NUM_EPOCHS*len(qc_array)*(len(min_array)-min_indx-1)+NUM_EPOCHS*len(qc_array)*len(min_array)*(len(layer_array)-layer_indx-1)))/60:.2f} min{min_b} nl{n_layer}')
                #print(f'T={T} Epoch: {epoch+1}/{NUM_EPOCHS} - Loss: {loss.item():.4f} b_loss={best_loss:.4f} - T: {time.time()-t0:.2f}s/epoch ,tempo_previto={(((NUM_EPOCHS-1-epoch+NUM_EPOCHS*(len(qc_array)-q_indx-1)+NUM_EPOCHS*len(qc_array)*(len(min_array)-min_indx-1)+NUM_EPOCHS*len(qc_array)*len(min_array)*(len(layer_array)-layer_indx-1)))):.2f} min{min_b} nl{n_layer}')
                plt.plot(loss_history)
                plt.savefig(f'loss__T{T}_nl{n_layer}_min{min_b}_qc{qc}_ancilla{Q_ANCILLA}_ld{ld_dim}.png')
                plt.close()
            np.save(f'thetas_T{T}_nl{n_layer}_min{min_b}_qc{qc}_{Q_ANCILLA}_ld{ld_dim}.npy',trained_thetas_1)
            np.save(f'loss__T{T}_nl{n_layer}_min{min_b}_qc{qc}_ancilla{Q_ANCILLA}_ld{ld_dim}.npy',loss_history)