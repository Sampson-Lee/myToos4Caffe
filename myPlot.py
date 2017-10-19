# -*- coding:utf-8 -*-
"""
Created on Thu Oct 19 10:49:15 2017

@author: Sampson
"""

import numpy as np  
import matplotlib.pyplot as plt  
import re

def parselog(log_file):
    with open(log_file, 'r') as log_file:
        log = log_file.read()
        
        # \d*\.\d+(e\d+)? 匹配浮点数，其中?匹配前一个字符0次或者1次，其中*匹配前一个字符0或者无限次，升级版 [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?
        floatpattern='[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
        
        # (?P<test_loss>...) 分组，除了原有的编号外再指定一个额外的别名，根据log文件具体信息修改   
        train_num_pattern = r"Iteration (?P<train_num>\d+) (\S+) iter/s"
        train_acc_pattern = r"Train net output #0: Accuracy1 = (?P<train_acc>" + floatpattern + ")"
        train_loss_pattern = r"Train net output #1: SoftmaxWithLoss1 = (?P<train_loss>" + floatpattern + ")"
        
        test_num_pattern = r"Iteration (?P<test_num>\d+), Testing net"
        test_acc_pattern = r"Test net output #0: Accuracy1 = (?P<test_acc>" + floatpattern + ")"
        test_loss_pattern = r"Test net output #1: SoftmaxWithLoss1 = (?P<test_loss>" + floatpattern + ")"
        
        train_losses = []
        train_accs = []
        train_iterations = []
        test_losses = []
        test_accs = []
        test_iterations = []
    
        for iter_num in re.findall(train_num_pattern, log):
            train_iterations.append(int(iter_num[0]))
    
        for loss in re.findall(train_loss_pattern, log):
            train_losses.append(float(loss[0]))
    
        for acc in re.findall(train_acc_pattern, log):
            train_accs.append(float(acc[0]))
            
        for iter_num in re.findall(test_num_pattern, log):
            test_iterations.append(int(iter_num))
    
        for loss in re.findall(test_loss_pattern, log):
            test_losses.append(float(loss[0]))
    
        for acc in re.findall(test_acc_pattern, log):
            test_accs.append(float(acc[0]))
            
        train_iterations = np.array(train_iterations)
        train_losses = np.array(train_losses)
        train_accs = np.array(train_accs)
        test_iterations = np.array(test_iterations[1:])
        test_losses = np.array(test_losses[1:])
        test_accs = np.array(test_accs[1:])
        
    return train_iterations,train_losses,train_accs,test_iterations,test_losses,test_accs

def parsetxt(flieName):  
    with open("./test.txt") as f:
        lines = f.readlines()
        names = lines.pop(0).split('#')[1].split()
        len1=len(lines)
        len2=len(names)
        data = np.zeros((len1,len2))
        for i,line in enumerate(lines):
            line=map(float, line.split()) # split total str and transform str to float
            data[i][:] = line[:]
            
    print('ok!')        
    return data,names
    
def plotData(data):  
    _, ax1 = plt.subplots()  
    ax2 = ax1.twinx()  
    # train loss -> green
    ax1.plot(data['t_i'], data['t_l'], 'g', label="t_l")
    # test loss -> yellow  
    ax1.plot(data['v_i'], data['v_l'], 'y', label="v_l")
    # test accuracy -> red  
    ax2.plot(data['t_i'], data['t_a'], 'r', label="t_a")
    # train accuracy ->  blue
    ax2.plot(data['v_i'], data['v_a'], 'b', label="v_a")
    
    ax1.set_xlabel('iteration')  
    ax1.set_ylabel('loss')  
    ax2.set_ylabel('accuracy')
    ax1.legend(loc='lower left')
    ax2.legend(loc='upper left')
    plt.savefig("./3.png")
    plt.show()

if __name__ == '__main__':
    log_file='./total.log'
    txtdir='./train.txt'
    data={}
    data['t_i'],data['t_l'],data['t_a'],data['v_i'],data['v_l'],data['v_a']= parselog(log_file)
    plotData(data)