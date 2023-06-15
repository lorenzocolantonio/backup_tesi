import numpy as np
import torch
import matplotlib.pyplot as plt

from Modules.pennylane_functions import *
from Modules.training_functions import *

n_samples=1000
T_array=np.array([5,10,25])
qc_array=np.array([0,64])
min_array=np.array([0.05,0.01,0.005])
layer_array=np.array([5,10,20])
count=0
for T in T_array:
     for n_layer in layer_array:
         for qc in qc_array:
            for min in min_array:
                all_thetas=np.load(f'ancilla_{Q_ANCILLA}/all_thetas_T{T}_nl{n_layer}_min{min}_qc{qc}_{Q_ANCILLA}_ld{ld_dim}.npy')
                
                thetas=all_thetas[24]
                print(np.shape(thetas))
                thetas=torch.tensor(thetas)
                quit()
                
    
            
                noise_batch = torch.randn(n_samples, ld_dim,2)
                noise_batch=torch.view_as_complex(noise_batch)
                
                zeros_tensor = torch.zeros(n_samples,  2**NUM_QUBITS-ld_dim)
                noise_batch=torch.cat([noise_batch, zeros_tensor], dim=1)
                

                #noise_batch =noise_batch.repeat(1,2)
                #noise_batch = noise_batch/torch.norm(noise_batch, dim = 1).view(-1, 1)
                # assign noise as value for first iteration of denoising loop
                denoised_batch = noise_batch
                to_save=[]

                    
                    # implement denoising loop

                for i in range(T):
                        #t=alphas_bar[T-i-1]*torch.ones(16)
                        # append first element of batch to hystory and loop
                        #to_save.append(torch.abs(denoised_batch).detach().numpy())
                        denoised_batch=denoised_batch/torch.norm(denoised_batch, dim = 1).view(-1, 1)
                        denoised_batch = circuit_aq(qc,thetas,n_layer, denoised_batch)
                        #enoised_batch = denoised_batch[:,:256].repeat(1,4) 
                        denoised_batch[:,ld_dim:]=0
                to_save.append(torch.abs(denoised_batch[:,:ld_dim]).detach().numpy())
                #print(np.shape(to_save))
                count+=1
                print(count/(len(T_array)*len(qc_array)*len(layer_array)*len(min_array)))
                np.save(f'all_generated_ld{ld_dim}_T{T}_nl{n_layer}_min{min}_qc{qc}_{Q_ANCILLA}.npy',to_save)
                '''# take amplitudes, rescale to be positive and normalise
                        denoised_batch_r = torch.real(denoised_batch)
                        denoised_batch_i = torch.imag(denoised_batch)
                        denoised_batch = torch.abs(denoised_batch)
                        denoised_batch = denoised_batch - torch.min(denoised_batch, dim = 1)[0].view(-1, 1)
                        #denoised_batch = denoised_batch/torch.norm(denoised_batch, dim = 1).view(-1, 1)
                        denoised_batch_r = denoised_batch_r - torch.min(denoised_batch_r, dim = 1)[0].view(-1, 1)
                        #denoised_batch_r = denoised_batch_r/torch.norm(denoised_batch_r, dim = 1).view(-1, 1)
                        denoised_batch_i = denoised_batch_i - torch.min(denoised_batch_i, dim = 1)[0].view(-1, 1)
                        #denoised_batch_i = denoised_batch_i/torch.norm(denoised_batch_i, dim = 1).view(-1, 1)

                        denoised_numpy=denoised_batch[:,:256].detach().numpy().reshape(-1,16, 16)
                        denoised_numpy_r=denoised_batch_r[:,:256].detach().numpy().reshape(-1,16, 16)
                        denoised_numpy_i=denoised_batch_i[:,:256].detach().numpy().reshape(-1,16, 16)'''

                    











