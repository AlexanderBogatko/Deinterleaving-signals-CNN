import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from math import sin

# const PRI signal defenition
def Const(N, Interval, TStart, BSize, numOfSignals):

    eps = BSize / 2

    Bin_Signal = np.zeros((N,), np.int8)

    start_bin = 0

    for i in range(N):
        if ( abs( i * BSize - TStart  ) < eps ):
            Bin_Signal[i] = 1
            start_bin = i
            break

    p = TStart + Interval

    sigma = 0.001 * Interval

    for i in range(1, numOfSignals):
        p = p + sigma * rand()

        for j in range (start_bin, N):
            if ( abs( j * BSize - p  ) < eps ):
                Bin_Signal[j] = 1
                start_bin = j
                break

        if (N - start_bin < 2):
            break

        p = p + Interval

    return Bin_Signal


def Wobul(N, Interval, TStart, BSize, numOfSignals):

	eps = BSize / 2

	Bin_Signal = np.zeros((N,), np.int8)

	start_bin = 0

	for i in range(N):
	    if ( abs( i * BSize - TStart  ) < eps ):
	        Bin_Signal[i] = 1
	        start_bin = i
	        break

	p = TStart + Interval

	sigma = 0.001 * Interval

	A = 0.05 * Interval

	omega = 50

	teta = 0.01

	for i in range(1, numOfSignals):
	    p = p + A * sin(omega * i + teta) + sigma * rand()

	    for j in range (start_bin, N):
	        if ( abs( j * BSize - p  ) < eps ):
	            Bin_Signal[j] = 1
	            start_bin = j
	            break

	    if (N - start_bin < 2):
	        break

	    p = p + Interval

	return Bin_Signal

# define lost pulses function
def lost_pulses(seq, percent):
    koeff = 10
    count = 0
    idxs = []
    for i in range(seq.shape[0]):
        if (seq[i] == 1):
            idxs.append(i)
            count = count + 1

    num_of_lost = int(percent * len(idxs))

    delete = np.zeros((num_of_lost,))
    n = 0
    while(n < num_of_lost):
        i = int(rand() * len(idxs))
        k = int(rand() * koeff)

        if( (i + k) >= len(idxs) ):
            k = 0

        
        for j in range(k):
            delete[n] = idxs[i + j]
            n = n + 1

            if(n >= num_of_lost):
                break
    
    for i in range(num_of_lost):
        seq[int(delete[i])] = 0

    return seq

# define spurious pulses function
def spur_pulses(seq, percent):
    koeff = 5
    count = 0
    idxs = []
    for i in range(seq.shape[0]):
        if (seq[i] == 1):
            idxs.append(i)
            count = count + 1

    num_of_spurious = int(percent * len(idxs))

    spr = []
    
    for i in range(num_of_spurious):
        spur_idx = int(rand() * len(idxs))

        
        idx = idxs[spur_idx] + 1 + int(koeff * rand())

        if (idx < seq.shape[0]):
            spr.append(idx)

    for i in range(len(spr)):
        seq[int(spr[i])] = 1

    return seq

def mix_two_signals(S1, S2):

    S = S1 + S2

    #for i in range(S1.shape[0]):
    #    if (S[i] > 1):
    #        S[i] = 1

    return np.array(S, np.float32)

def mix_signals(Signals):

    num_of_signals = Signals.shape[0]

    if (num_of_signals == 1):
        return Signals
    else:
        if (num_of_signals == 2):
            return mix_two_signals(Signals[0], Signals[1])
        else:
            result = mix_two_signals(Signals[0], Signals[1])

            for i in range(2, num_of_signals):
                result = mix_two_signals(result, Signals[i])
                
            return result
        

def show_test_signals(Signals, BSize):
    PRIs = []
    TOAs = []
    for sig in Signals:
        TMP = []
        for i in range(sig.shape[0]):
            if (sig[i] == 1):
                TMP.append(i)
        Hist = []
        TOA = []
        for i in range(1, len(TMP)):
            Hist.append((TMP[i] - TMP[i - 1]))
            TOA.append(TMP[i] * BSize)
        
        PRIs.append(Hist)
        TOAs.append(TOA)

    for pri, toa in zip(PRIs, TOAs):
        plt.scatter(np.array(toa), np.array(pri), c='b', marker='o', s=1)
    #plt.ylabel('probability')
    plt.xlim([0, 1 * 10 ** (-2)])
    plt.ylim([0, 100])
    plt.ylabel('PRI')
    plt.xlabel('Time of arrival')
    plt.show()




def make_bin_signal(f_idx, p, dt, w, tau):
    signal = np.zeros((200, 500), np.int8)
    
    k = 0
    br = False
    for i in range(signal[f_idx].shape[0]):
        for j in range(tau):
            if(k + j < signal[f_idx].shape[0]):
                signal[f_idx][k + j] = 1
            else:
                br = True
                break
        if(br):
            break
        if(w):
            k = k + int(p / dt) + int(rand() * 0.4 * p / dt)
        else:
            k = k + int(p / dt)
        if (k >= signal[f_idx].shape[0]):
            break
    return signal


def mix_s(signals):
    res = np.zeros((signals.shape[1], signals.shape[2]), np.float32)
    for s in signals:
        res = res + s
    return res / np.amax(res)
    

