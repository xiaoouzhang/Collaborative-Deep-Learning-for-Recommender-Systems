#calculate the percentile-ranking of new products for both old and new users.

import numpy as np
#import pandas as pd
import h5py
from scipy import stats
import auto_fun as auto
alpha=40

#load training and validation subset
with h5py.File('rating_tr_numpy.h5', 'r') as hf:
    rating_tr = hf['rating'][:].astype(int)


with h5py.File('rating_val_numpy.h5', 'r') as hf:
    rating_val = hf['rating'][:].astype(int)

#print(rating_tr)

#load u and v
with h5py.File('u_40_40+100.h5', 'r') as hf:
#with h5py.File('u.h5', 'r') as hf:
    u = hf['u'][:]
with h5py.File('v_40_40+100.h5', 'r') as hf:
#with h5py.File('v.h5', 'r') as hf:
    v = hf['v'][:]
'''
with h5py.File('u_2_mono_40+10_auto.h5', 'r') as hf:
#with h5py.File('u.h5', 'r') as hf:
    u = hf['u'][:]
with h5py.File('v_2_mono_40+10_auto.h5', 'r') as hf:
#with h5py.File('v.h5', 'r') as hf:
    v = hf['v'][:]
with h5py.File('W1_2_mono_40+10.h5', 'r') as hf:
    W1 = hf['W1'][:]
with h5py.File('b1_2_mono_40+10.h5', 'r') as hf:
    b1 = hf['b1'][:]
with h5py.File('c1_2_mono_40+10.h5', 'r') as hf:
    c1 = hf['c1'][:]
'''
'''
with h5py.File('W2_20_30bi_40+100_10.h5', 'r') as hf:
    W2 = hf['W2'][:]
with h5py.File('b2_20_30bi_40+100_10.h5', 'r') as hf:
    b2 = hf['b2'][:]
with h5py.File('c2_20_30bi_40+100_10.h5', 'r') as hf:
    c2 = hf['c2'][:]
'''
#print(u)
#print(v)

#preference and confidence
p=np.zeros(rating_tr.shape)
p[rating_tr>0]=1
c=np.zeros(rating_tr.shape)
c=1+alpha*rating_tr

print(np.linalg.norm(p-np.dot(u,v.T)))
#print(p-np.dot(u,v.T))

#only retain the new choices
rating_val[rating_tr>0]=0

r_pred=np.dot(u,v.T)

#set mask
m=(p>0)
#r_masked=np.ma.masked_array(r_pred, mask=m)
#top=np.zeros(rating_tr.shape[0])
#top=r_masked.argmax(axis=1)

#predict
#correct=0
rank=0
total=0
for i in range(rating_val.shape[0]):
    prod=rating_val[i]
    prod_predict=np.ma.masked_array(r_pred[i],mask=m[i])
    if np.sum(prod)>0:
        for j in range(prod.size):
            if prod[j]>0:
                total=total+1
                rank=rank+stats.percentileofscore(prod_predict[~prod_predict.mask],prod_predict[j])
        #total=total+1
        #if rating_val[i,top[i]]>0:
        #    correct=correct+1
print(total)
print(100-rank/total)
'''
print('new users')

diction = [('ind_empleado', 5), ('pais_residencia', 24), ('sexo', 3), ('ind_nuevo', 2), ('indrel', 2), ('indrel_1mes', 4), ('tiprel_1mes', 4), ('indresi', 2), ('indext', 2), ('conyuemp', 3), ('canal_entrada', 158), ('indfall', 2), ('cod_prov', 53), ('ind_actividad_cliente', 2), ('segmento', 4), ('antiguedad_binned', 10), ('age_binned', 24), ('renta_binned', 10)]

lenList = []
for tuppl in diction:
    val = tuppl[1]
    lenList.append(val)
accList = []
for i in range(len(lenList)):
    if i ==0:
        accList.append(lenList[i])
    else:
        accList.append(accList[i-1]+lenList[i])
#read user infor
with h5py.File('user_infor_new.h5', 'r') as hf:
    x_new = hf['infor'][:]
#read rating matrix
with h5py.File('rating_tr_numpy_new.h5', 'r') as hf:
    rating_tr = hf['rating'][:].astype(int)
with h5py.File('rating_val_numpy_new.h5', 'r') as hf:
    rating_val = hf['rating'][:].astype(int)


#u =  auto.getoutPut(W1,W2,b1,b2,x_new,accList)
u = auto.getoutPut_mono(W1,b1,x_new,accList)
#u = np.random.rand(u.shape[0],u.shape[1])
#v = np.random.rand(v.shape[0],v.shape[1])
#preference and confidence
p=np.zeros(rating_tr.shape)
p[rating_tr>0]=1
c=np.zeros(rating_tr.shape)
c=1+alpha*rating_tr

print(np.linalg.norm(p-np.dot(u,v.T)))
#print(p-np.dot(u,v.T))

#only retain the new choices
rating_val[rating_tr>0]=0

r_pred=np.dot(u,v.T)

#set mask
m=(p>0)
#r_masked=np.ma.masked_array(r_pred, mask=m)
#top=np.zeros(rating_tr.shape[0])
#top=r_masked.argmax(axis=1)

#predict
#correct=0
rank=0
total=0
for i in range(rating_val.shape[0]):
    prod=rating_val[i]
    prod_predict=np.ma.masked_array(r_pred[i],mask=m[i])
    if np.sum(prod)>0:
        for j in range(prod.size):
            if prod[j]>0:
                total=total+1
                rank=rank+stats.percentileofscore(prod_predict[~prod_predict.mask],prod_predict[j])
        #total=total+1
        #if rating_val[i,top[i]]>0:
        #    correct=correct+1
print(total)
print(100-rank/total)
'''
