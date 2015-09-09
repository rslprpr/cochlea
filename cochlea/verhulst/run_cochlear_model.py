from __future__ import division
import sys

import pandas as pds

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
### % Verhulst cochlea model address
sys.path.append('/Downloads/cochlea-master/verhulst')

import cochlear_model

Fs=96000
f0=1000
tmax=0.05
t=np.arange(0, tmax, 1./Fs)
insig=np.zeros((t.size,2))
insig[:,0]=np.sin(2*math.pi*f0*t )
insig[:,1]=insig[:,0]
insig = insig
fc='all'
Oversampling=1
sectionsNo=1000
p0=float(2e-5)
probe_points=fc
spl=np.array((60,60))
channels, idx= insig.T.shape
subjectNo=0
norm_factor=p0*10.**(spl/20.)
sheraPo=0.06
irr_on=np.array([0,1])   

### Normalizing
sig = (insig * norm_factor).T

### Linear Chirp Signal, make sound with frequency varies linearly with time 
chirp_sig = signal.chirp(t, 80, t[-1], 20000) 

### Run the model  
cochlear_list=[ [cochlear_model.cochlea_model(),
                 sig[i],
                 irr_on[i],
                 i] for i in range(channels)]# channels=0,1

### definition here, to have all the parameter implicit
def solve_one_cochlea(model): 
    i=model[3]
    coch=model[0]
    coch.init_model(model[1],
                    Oversampling*Fs,
                    sectionsNo,
                    probe_points,
                    Zweig_irregularities=model[2],
                    sheraPo=sheraPo,
                    subject=subjectNo) 
    coch.solve()
    f=open("out/v"+str(i+1)+".np",'wb')
    np.array(coch.Vsolution,dtype='=d').tofile(f)
    f.close()
    f=open("out/y"+str(i+1)+".np",'wb')
    np.array(coch.Ysolution,dtype='=d').tofile(f)
    f.close()
    f=open("out/E"+str(i+1)+".np",'wb')
    np.array(coch.oto_emission,dtype='=d').tofile(f)
    f.close()
    f=open("out/F"+str(i+1)+".np",'wb')
    np.array(coch.cf,dtype='=d').tofile(f)
    f.close()

    ### plot the results
    fig = plt.figure(figsize=(13, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.suptitle('results')

    
    ### Chirp Spectrogram 
    ax = fig.add_subplot(322)
    plt.specgram(chirp_sig,Fs=48000,cmap='RdBu')
    plt.xlim(0.00, 0.08)
    plt.ylim(0, 30000)
    plt.xlabel('Time[ms]')
    plt.ylabel('Frequency[Hz]')
    plt.title("Chirp")
    
    ax = fig.add_subplot(323)
    img2=ax.imshow(coch.Ysolution.T, cmap='RdBu')
    ax.set_xlim(200, 1000)
    ax.set_ylim(0, 700)
    plt.xlabel('Time[ms]')
    plt.title("Basilar membrane displacement")
    plt.colorbar(img2)

    ax = fig.add_subplot(321)
    img=ax.imshow(coch.Vsolution.T,cmap='RdBu')
    ax.set_xlim(200, 1000)
    ax.set_ylim(0, 700)
    plt.xlabel('Time[ms]')
    plt.title("Basilar membrane velocity")
    plt.colorbar(img)
    
    ax = fig.add_subplot(324)
    plt.plot(coch.Vsolution[-10,:])    
    plt.title("Basilar membrane displacement")
    
    ax = fig.add_subplot(326)    
    img=plt.plot(coch.cf)
    plt.title("Center Frequencies")
    plt.show()    
    
    ### Dictionary which calculate the center frequencies over 20e3 Hz  
#    vlist=[v for v in coch.Vsolution.T]    
#    ylist=[y for y in coch.Ysolution.T]
#    df=pds.DataFrame({'Cf':coch.cf[1:], 'Y':ylist, 'V':vlist})
#    return df[df['Cf']>10e3]