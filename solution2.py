# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:32:18 2020

@author: admin
"""

import pandas as pd
from itertools import combinations 
  
def getpair(mylist, k): 
    return [(i, j) for i, j in combinations(mylist, r = 2) 
                   if abs(i - j) == k] 
              
# Driver code 
lst = [1,3,5] 
k = 2
outpairs = getpair(lst,k)

print("we will have",len(outpairs),"pairs:", outpairs) 
