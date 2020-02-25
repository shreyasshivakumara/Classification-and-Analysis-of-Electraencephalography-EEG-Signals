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
from sklearn.linear_model import LogisticRegression

howmany=500;




with open('classdata.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    i=0;
    for row in readCSV:
        cdata=np.asarray(row);

    
my_in=[]
my_out=[]
test_in=[]
test_out=[]

bmy_in=[]
btest_in=[]
temp_in=[]
temp_out=[]
btemp_in=[]



with open('feature.txt') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    ci=0;
    for row in readCSV:
        data=np.asarray(row);
        
        temp_in.append(float(data[0]))
        btemp_in.append(float(data[2]));
        temp_in.append(float(data[1]))        
        temp_in.append(float(data[2]))
        temp_in.append(float(data[3]))
        temp_in.append(float(data[4]))
        temp_in.append(float(data[5]))
        temp_out.append(int(cdata[ci]))
          
        if howmany>=0:
            my_in.append(temp_in)
            my_out.append(temp_out)
            bmy_in.append(btemp_in)
        else:
            test_in.append(temp_in)
            test_out.append(temp_out)
            btest_in.append(btemp_in)
        temp_in=[]
        temp_out=[]
        btemp_in=[]
       
        ci=ci+1;
        howmany=howmany-1;

reg= LogisticRegression()
reg.fit(bmy_in,my_out);

gnb = GaussianNB()
gnb.fit(my_in,my_out)

pred_test=gnb.predict(test_in);
pred_test1=reg.predict(btest_in);

sz=len(pred_test);
acc=0;
acc2=0;
for i in range(0,sz-1):
    if pred_test[i]==test_out[i]:
        acc=acc+1
    if pred_test1[i]==test_out[i]:
        acc2=acc2+1    

accper=acc*100.0/sz;
accper2=acc2*100.0/sz;
print('Enhanced Accuracy is ',accper);
print('Base Accuracy is ',accper2);

msevalues=[]
msevalues.append(accper);
msevalues.append(accper2);


objects = ('Proposed', 'Base')
y_pos = np.arange(len(objects))
plt.bar(y_pos, msevalues, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Error Graph')
plt.show()


        
