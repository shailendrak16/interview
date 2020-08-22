# -*- coding: utf-8 -*-
"""
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
