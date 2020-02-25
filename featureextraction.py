import csv
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from numpy import *
from scipy.signal import *
from numpy.fft import * 
from matplotlib import *
from scipy import *
from pylab import *
import tsfresh
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

outfile = open("feature.txt","w")





tottoread=700;



with open('traindata.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    ci=0;
    for row in readCSV:
        #while i<2:
        #    i=i+1
        #    continue
        #print(row)
        # Define sampling frequency and time vector
        data=np.asarray(row);
        sf = 100.
        time = np.arange(data.size) / sf

        # Plot the signal
        #fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        #plt.plot(time, data, lw=1.5, color='k')
        #plt.xlabel('Time (seconds)')
        #plt.ylabel('Voltage')
        #plt.xlim([time.min(), time.max()])
        #plt.title('N3 sleep EEG data (F3)')
        #sns.despine()
        #plt.show();
        y=data.astype(float);
        L = len(y)            # signal length
        fs = 100.0              # sampling rate
        T = 1/fs                # sample time
        t= linspace(1,L,L)*T   # time vector

        f = fs*linspace(0,L/10,L/10)/L  # single side frequency vector, real frequency up to fs/2
        Y = fft(y)
        filtered = []
        b= [] # store filter coefficient
        cutoff = [0.5,4.0,7.0,12.0,30.0]

        for band in range(0, len(cutoff)-1):
            wl = 2*cutoff[band]/fs*pi
            wh = 2*cutoff[band+1]/fs*pi
            M = 512      # Set number of weights as 128
            bn = zeros(M)
         
            for i in range(0,M):     # Generate bandpass weighting function
                n = i-  M/2       # Make symmetrical
                if n == 0:
                   bn[i] = wh/pi - wl/pi;
                else:
                   bn[i] = (sin(wh*n))/(pi*n) - (sin(wl*n))/(pi*n)   # Filter impulse response
         
            bn = bn*kaiser(M,5.2)  # apply Kaiser window, alpha= 5.2
            b.append(bn)
        [w,h]=freqz(bn,1)
        filtered.append(convolve(bn, y)) # filter the signal by convolving the signal with filter coefficients
        #figure(figsize=[16, 10])
        #subplot(2, 1, 1)
        #plot(y)
        for i in range(0, len(filtered)):
            y_p = filtered[i]
            #plot(y_p[ M//2:L+M//2])
            if i==3:
                alphasig=y_p;
        #axis('tight')
        #title('Time domain')
        #xlabel('Time (seconds)')
        #show();
        
        p1=tsfresh.feature_extraction.feature_calculators.abs_energy(y_p)
        #print(p1)
        p2=tsfresh.feature_extraction.feature_calculators.absolute_sum_of_changes(y_p)
        #print(p2)
        p3=tsfresh.feature_extraction.feature_calculators.kurtosis(y_p);
        #print(p3)
        p4=tsfresh.feature_extraction.feature_calculators.mean_change(y_p);
        #print(p4)
        p5=tsfresh.feature_extraction.feature_calculators.mean_second_derivative_central(y_p);
        #print(p5)
        p6=tsfresh.feature_extraction.feature_calculators.sample_entropy(y_p);
        #print(p6)
        #p7=tsfresh.feature_extraction.feature_calculators.skewnes(y_p);
        #print(p7)
        line=str(p1) + "," + str(p2) + ","+ str(p3) +"," + str(p4) + "," + str(p5) + "," + str(p6);
        print(line)
        outfile.write(line);
        outfile.write("\n");
      
        tottoread=tottoread-1;
        if tottoread==0:
            break;
       
outfile.close()







