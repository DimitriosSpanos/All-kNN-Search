# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 17:39:42 2021

@author: Dimitris Spanos
"""

import matplotlib.pyplot as plt
import numpy as np



def make_plot(name,n,d,k):
    
    x2, y2 = np.loadtxt('V0results.txt', delimiter=',', unpack=True)
    plt.plot(x2,y2, label='V0')
    x3, y3 = np.loadtxt('V1results.txt', delimiter=',', unpack=True)
    plt.plot(x3,y3, label='V1')
    x4, y4 = np.loadtxt('V2results.txt', delimiter=',', unpack=True)
    plt.plot(x4,y4, label='V2')

    plt.xlabel('Number of Processors')
    plt.ylabel('Execution Time(sec)')
    plt.title('%s: (n, d, k) = (%s, %s, %s)' % (name,n, d, k))
    plt.rcParams["figure.figsize"] = (8,8)

    plt.legend()
    plt.savefig('re.png', bbox_inches='tight')
    plt.show()    
    
name = input("name: ")
n = input("n: ")
d = input("d: ")
k = input("k: ")

make_plot(name, n, d, k)