import h5py
import numpy as np
from numpy.random import rand, choice
from common import Const, Wobul, mix_signals, lost_pulses, spur_pulses


# define data

bin_size = 1 * 10 ** (-6)

minPRI = 1 * 10 ** (-5)

maxPRI = 9 * 10 ** (-5)

h_PRI = 0.1 * 10 ** (-5)

PRIx = int((maxPRI - minPRI) / h_PRI)

PRI_HIST = np.zeros((PRIx + 1,), np.float32)

for i in range(PRIx + 1):
    PRI_HIST[i] = minPRI + i * h_PRI

Time_Interval = 1 * 10 ** (-2)

N = int(Time_Interval / bin_size)

Max_Number_Of_Signals = 5   # number of signals in mixture

Max_T_Start = 0.6 * 10 ** (-2)

h_T = 0.05 * 10 ** (-2)     # timestep

Tx = int(Max_T_Start / h_T)

Start_Times = np.zeros((Tx,), np.float32)

for i in range(Tx):
    Start_Times[i] = i * h_T

Max_LostPulses_Percent = 0.1
Max_SpuriousPulses_Percent = 0.15

numOfSignals = 1000

NumOfDataForEachPRI = 5

modulationTypes = np.array([0, 1]) # 0 - const, 1 - wobulated

# generate train set
print 'generating test data...'

Separate_Signals = np.zeros( (PRI_HIST.shape[0] * NumOfDataForEachPRI, Max_Number_Of_Signals, N) )
X = np.zeros((PRI_HIST.shape[0] * NumOfDataForEachPRI, N), np.float32)
Y = np.zeros((PRI_HIST.shape[0] * NumOfDataForEachPRI, modulationTypes.shape[0], PRI_HIST.shape[0]), np.int8)

k = 0
for i in range(PRI_HIST.shape[0]):
    for n in range(NumOfDataForEachPRI):

        sig_count = int(Max_Number_Of_Signals * rand()) + 1

        y = np.zeros((modulationTypes.shape[0], PRI_HIST.shape[0]), np.int8)

        Signals = np.zeros((Max_Number_Of_Signals, N), np.int8)

        for j in range(sig_count):
            T_Start = Start_Times[ int(Start_Times.shape[0] * rand()) ]

            if (j == 1):
                modul_type = choice(modulationTypes)
                if (modul_type == 0):
                    Signal = Const(N, PRI_HIST[i], T_Start, bin_size, numOfSignals)
                else:
                    Signal = Wobul(N, PRI_HIST[i], T_Start, bin_size, numOfSignals)
                y[modul_type][i] = 1
            else:
                t = int( PRI_HIST.shape[0] * rand() )
                modul_type = choice(modulationTypes)
                if (modul_type == 0):
                    Signal = Const(N, PRI_HIST[t], T_Start, bin_size, numOfSignals)
                else:
                    Signal = Wobul(N, PRI_HIST[t], T_Start, bin_size, numOfSignals)
                y[modul_type][t] = 1

            Signal = lost_pulses(Signal, Max_LostPulses_Percent * rand())

            Signal = spur_pulses(Signal, Max_SpuriousPulses_Percent * rand())

            Signals[j] = Signal

        Separate_Signals[k] = Signals
        mixSignal = mix_signals(Signals)
        X[k] = mixSignal / np.amax(mixSignal)
        Y[k] = y
        print 'sample ' + str(k) + ' from ' + str(X.shape[0])
        k = k + 1

print X.shape
print Y.shape

            
# write train data
print 'writing test data...'

f = h5py.File('TestDataModulation.hdf5', 'w')

X = f.create_dataset('X', data=X)
Y = f.create_dataset('Y', data=Y)
Separate_Signals = f.create_dataset('S', data=Separate_Signals)
PRI_HIST = f.create_dataset('PRI_HIST', data=PRI_HIST)

print 'test data is ready!'
