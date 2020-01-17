# -*- coding: utf-8 -*-
"""
Python code of Gravitational Search Algorithm (GSA)
Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
           Information sciences 179.13 (2009): 2232-2248.	
Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/7ossam81/EvoloPy and matlab version of GSA at mathworks.

Purpose: Main file of Gravitational Search Algorithm(GSA) 
            for minimizing of the Objective Function

Code compatible:
 -- Python: 2.* or 3.*
"""

import random
import numpy
import math
from solution import solution
import time
import massCalculation
import gConstant
import gField
import move
import benchmarks
import numpy as np

def Initialization(dim,N,up,down):
    temp = np.random.rand(N,dim)
    if up.shape[1] == 1:
        temp *= (up-down)+down
        
    if up.shape[1] >1 :
        for i in range(dim):
            high = up[0,i]

            low = down[0,i]
            temp[:,i] = np.add(np.multiply(temp[:,i], (high-low)),low)
            
    x = temp.astype(int)

    return x
        
def GSA(lb,ub,dim,PopSize,iters,pxy):
    
    ElitistCheck =1
    Rpower = 1 
    
    s=solution()
        
    N = PopSize
    """ Initializations """
    
    vel=numpy.zeros((PopSize,dim))
    fit = numpy.zeros(PopSize)
    M = numpy.zeros(PopSize)
    gBest=numpy.zeros(dim)
    gBestScore=float("inf")
    '''
    pos=numpy.random.uniform(0,1,(PopSize,dim)) #*(ub-lb)+lb

    ub = ub.reshape(pos.shape)
    lb = lb.reshape(pos.shape)

    pos = pos *(ub-lb)+lb
    '''
    up = ub
    low = lb
    '''
    dim1=dim;
    X1= Initialization(dim1,N,up,low) 
    for si in range(len(X1)):
           X1[si,:]=sorted(X1[si,:]); 

    dim2=dim;
    X2= Initialization(dim2,N,up,low) 
    for si in range(len(X2)):
           X2[si,:]=sorted(X2[si,:])


    X_org= np.concatenate((X1,X2), axis =1)

    Xtemp1=X_org[:,1:dim]
    for si in range(len(Xtemp1)):
        Xtemp1[si,:]=sorted(Xtemp1[si,:])

    thdim=2*dim

    Xtemp2=X_org[:,(dim+1):thdim]
    for si in range(len(Xtemp2)):
        Xtemp2[si,:]=sorted(Xtemp2[si,:])

    X=np.concatenate((Xtemp1,Xtemp2),axis=1)
    '''
    
    X = []
    for _ in range(0,PopSize):
        a = np.random.randint(low = 0, high = 255, size = dim)
        b = np.random.randint(low = 0, high = 255, size = dim)
        a = np.sort(a)
        b = np.sort(b)
        temp = np.concatenate((a,b),axis=None)
        X.insert(0,list(temp))
    
    X = np.array(X)
    convergence_curve=numpy.zeros(iters)
    
    print("GSA is optimizing  \""+"Renyi entropy"+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    for l in range(0,iters):
        for i in range(0,PopSize):
            l1 = X[i:]
            

            #Calculate objective function for each particle
            fitness=[]
            fitness=benchmarks.entropy(l1,pxy)
            fit[i]=fitness
    
                
            if(gBestScore>fitness):
                gBestScore=fitness
                gBest=l1           
        
        """ Calculating Mass """
        M = massCalculation.massCalculation(fit,PopSize,M)

        """ Calculating Gravitational Constant """        
        G = gConstant.gConstant(l,iters)        
        
        """ Calculating Gfield """        
        acc = gField.gField(PopSize,dim,X,M,l,iters,G,ElitistCheck,Rpower)
        
        """ Calculating Position """        
        X, vel = move.move(PopSize,dim,X,vel,acc)
        
        convergence_curve[l]=gBestScore
      
        if (l%100==0):
               print(['At iteration '+ str(l)+ ' the best fitness is '+ str(gBestScore)])
               print('At iteration '+ str(l)+ ' the gBest valve '+ str(gBest))
               #print()
               
    timerEnd=time.time()  
    gBestIndividual = np.mean(gBest,axis= 0).astype(int)
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.Algorithm="GSA"
    s.objectivefunc="Renyi entropy"
    s.bestIndividual = gBestIndividual


    return s
         
    
