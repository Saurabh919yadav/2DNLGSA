# -*- coding: utf-8 -*-
"""
Python code of Gravitational Search Algorithm (GSA)
Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
           Information sciences 179.13 (2009): 2232-2248.	

Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/7ossam81/EvoloPy and matlab version of GSA at mathworks.

 -- Purpose: Defining the benchmark function code 
              and its parameters: function Name, lowerbound, upperbound, dimensions

Code compatible:
 -- Python: 2.* or 3.*
"""

import numpy
import math


    
def entropy(x,pxy):
	M = x.shape[0]
	N = x.shape[1]
	alpha = 0.5
	pxy[pxy == 0] = numpy.finfo(numpy.float).eps



	s1 = x[M-1,0]
	s2 = x[M-1,1]
	# s3 = x[M-1,2]
	# s4 = x[M-1,3]

	t1 = x[M-1,2]
	t2 = x[M-1,3]
	# t3 = x[M-1,6]
	# t4 = x[M-1,7]

	s1 = math.ceil(s1)
	s2 = math.ceil(s2)
	# s3 = math.ceil(s3)
	# s4 = math.ceil(s4)

	t1 = math.ceil(t1)
	t2 = math.ceil(t2)
	# t3 = math.ceil(t3)
	# t4 = math.ceil(t4)

	p0=0

	for i in range(s1+1):
		for j in range(t1+1):
			p0 += pxy[i][j]


	p1=0 
	for i in range(s1+1,s2+1):
		for j in range(t1+1,t2+1):
			p1 += pxy[i][j]
	'''
	p2=0
	for i in range(s2+1,s3+1):
		for j in range(t2+1,t3+1):
			p2 += pxy[i][j]


	p3=0
	for i in range(s3+1,s4+1):
		for j in range(t3+1,t4+1):
			p3 += pxy[i][j]
	

	p4 = 0
	for i in range(s4+1,256):
		for j in range(t4+1,256):
			p4 += pxy[i][j]
	'''
	h0=0

	for i in range(s1+1):
		for j in range(t1+1):
			sd=(pxy[i][j]/p0)**alpha
			h0 += sd

	h1=0
	for i in range(s1+1,s2+1):
		for j in range(t1+1,t2+1):
			sd=(pxy[i][j]/p1)**alpha
			h1 += sd
	'''
	h2=0
	for i in range(s2+1,s3+1):
		for j in range(t2+1,t3+1):
			sd=(pxy[i][j]/p2)**alpha
			h2 += sd

	h3=0
	for i in range(s3+1,s4+1):
		for j in range(t1+1,t4+1):
			sd=(pxy[i][j]/p3)**alpha
			h3 += sd

	h4=0
	for i in range(s4+1,256):
		for j in range(t4+1,256):
			sd=(pxy[i][j]/p4)**alpha
			h4 += sd
	'''
	h0 = (1/(1-alpha))*numpy.log(h0+numpy.finfo(numpy.float).eps)

	h1 = (1/(1-alpha))*numpy.log(h1+numpy.finfo(numpy.float).eps)
	# h2 = (1/(1-alpha))*numpy.log(h2+numpy.finfo(numpy.float).eps)
	# h3 = (1/(1-alpha))*numpy.log(h3+numpy.finfo(numpy.float).eps)
	# h4 = (1/(1-alpha))*numpy.log(h4+numpy.finfo(numpy.float).eps)

	# h = h0+h1+h2+h3+h4
	h = h0+h1

	return h








def getFunctionDetails(a):
  # [name, lb, ub, dim]
  param = {  0: ["entropy",-100,100,2],
            }
  return param.get(a, "nothing")



