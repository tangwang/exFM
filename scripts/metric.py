import sys
import logging
import traceback
import cgitb
import os
from optparse import OptionParser

test_file = sys.argv[1]
pred_file = sys.argv[2]

def calAUC(prob,labels):
    f = list(zip(prob,labels))
    rank = [values2 for values1,values2 in sorted(f,key=lambda x:x[0])]
    rankList = [i+1 for i in range(len(rank)) if rank[i]==1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if(labels[i]==1):
            posNum+=1
        else:
            negNum+=1
    auc = 0
    
    auc = float(sum(rankList)- (posNum*(posNum+1))/2)/(posNum*negNum)
    return auc

y = []
pred = []

with open(pred_file, 'r') as infile:
    for line in infile:
        pred.append(float(line.strip().split(' ')[1]))
with open(test_file, 'r') as infile:
    for line in infile:
        y.append(int(line.strip().split(' ')[0]))
print(len(y))
print(len(pred))
print(calAUC(pred,y))
