# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: admin

"""

import pandas as pd
import numpy as np 

# added some extra datapoint
data = [
("username1","phone_number1", "email1"),
("usernameX","phone_number1", "emailX"),
("usernameZ","phone_numberZ", "email1Z"),
("usernameY","phone_numberY", "emailX"),
("username2","phone_number2", "email2"),
("username3","phone_number3", "email1"),
]
  
#defining function which can retunr all common elements of the 2 list
def common (list1, list2):
   list_common = [i for i in list1 if  i in list2]
   return list_common

maxlength = len(data) 
commonindex1 = [0]
commonindex2 = []

#loop for getting indices for each row which have common elements of other rows
for i in range(maxlength):
   commonindex1 = [i] 
   for j in range(i+1,(maxlength)):
           z=common(data[i],data[j])
           if z:
               commonindex1.append(j)
   commonindex2.append(commonindex1)         


# merging all the list of indices got from above loop to get final list of indices with common elements
indices = []
while len(commonindex2)>0:
    initial, *rest = commonindex2
    initial = set(initial)
    k = -1
    while len(initial)> k:
        k = len(initial)

        rest2 = []
        for r in rest:
            if len(initial.intersection(set(r)))>0:
                initial |= set(r)
            else:
                rest2.append(r)     
        rest = rest2

    indices.append(initial)
    commonindex2 = rest

print(indices)






