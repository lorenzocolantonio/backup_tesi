import numpy as np
import matplotlib.pyplot as plt

from Modules.pennylane_functions import *
from Modules.training_functions import *
T_arr=np.array([5,10,25])
ld_dim_arr=np.array([4])
qc_array=np.array([32,48,56,60,62])
min_array=np.array([0.1,0.05,0.01,0.08])
layer_array=np.array([5,10,20,50])
count=0
ld_dim=4
for t_indx in range(len(T_arr)):
    T=T_arr[t_indx]
    for layer_indx in range(len(layer_array)):
        n_layer=layer_array[layer_indx]
        for q_indx in range(len(qc_array)):
            qc=qc_array[q_indx]
            for min_indx in range(len(min_array)):
                min_b=min_array[min_indx]
                loss=np.load(f'loss__T{T}_nl{n_layer}_min{min_b}_qc{qc}_ancilla{4}_ld{ld_dim}.npy')
                plt.figure()
                plt.plot(loss)
                plt.savefig(f'loss__T{T}_nl{n_layer}_min{min_b}_qc{qc}_complex_ancilla{Q_ANCILLA}.png')
                
                count+=1
                print(f'{count}/{len(T_arr)*len(qc_array)*len(min_array)*len(layer_array)}')