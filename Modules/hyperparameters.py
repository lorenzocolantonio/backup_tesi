import numpy as np
Q_ANCILLA=4
NUM_QUBITS =   5+Q_ANCILLA
ld_dim=32
NUM_LAYER_ENCODER=60
NUM_LAYER_DECODER=60
NUM_LAYER_LT=2
Q_COMPRESSION=0
LEARNING_RATE = 0.005
BATCH_SIZE = 256
NUM_EPOCHS = 40
SCHEDULER_PATIENCE = 5
SCHEDULER_GAMMA = 0.5
T = 5
to_zero=np.array([128,128+64,128+64+32,128+64+32+16])
STD=0.1
Q_tozero=0