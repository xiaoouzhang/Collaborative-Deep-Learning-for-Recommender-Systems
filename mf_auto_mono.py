#hybride model of SDAE and MF. One hidden layer is used in the SDAE.

import h5py
import numpy as np
import random
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.style.use('ggplot')
#import pandas as pd
import auto_fun as auto

INPUT_LAYER = 314
HIDDEN_UNIT1 = 40
HIDDEN_UNIT2 = 40
LEARNING_RATE = 0.001/50
EPOCH_NUM = 50
#randomSeed = np.random.RandomState(42)
mu, sigma = 0, 0.1
l=40
alpha=40
l2_u=100.0
l2_v=100.0
batch=500
ratio_l=300.0
ratio_u=1.0

def main(denoise = True):
    #allMatrix, xtrain, xval, xtest,lenList,accList = getData()
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
    with h5py.File('user_infor.h5', 'r') as hf:
        xtrain = hf['infor'][:]
    #read rating matrix
    with h5py.File('rating_tr_numpy.h5', 'r') as hf:
        rating_mat = hf['rating'][:]
    
    W1,W2,b1,b2,c1,c2 = auto.initialization(INPUT_LAYER,HIDDEN_UNIT1,HIDDEN_UNIT2,mu,sigma)
    #define user and item matrices
    u=np.random.rand(rating_mat.shape[0],l)
    v=np.random.rand(rating_mat.shape[1],l)

    #with h5py.File('u_40_mono_40+100_auto.h5', 'r') as hf:
    #    u = hf['u'][:]
    #with h5py.File('v_40_mono_40+100_auto.h5', 'r') as hf:
    #    v = hf['v'][:]
    #with h5py.File('W1_40_mono_40+100.h5', 'r') as hf:
    #    W1 = hf['W1'][:]
    #with h5py.File('b1_40_mono_40+100.h5', 'r') as hf:
    #    b1 = hf['b1'][:]
    #with h5py.File('c1_40_mono_40+100.h5', 'r') as hf:
    #    c1 = hf['c1'][:]

    #define preference and confidence matrices
    p=np.zeros(rating_mat.shape)
    p[rating_mat>0]=1
    c=np.zeros(rating_mat.shape)
    c=1+alpha*rating_mat

    iteration=30

    print('start')
    for iterate in range(iteration):
        #update u
        
        for i in range(rating_mat.shape[0]):
            c_diag=np.diag(c[i,:])
            temp_u=np.dot(np.dot(p[i,:],c_diag),v)
            u[i,:]=np.dot(temp_u,np.linalg.pinv(l2_u*np.identity(l)+np.dot(np.dot(v.T,c_diag),v)))
        print('u complete')
        
        #update v
        for j in range(rating_mat.shape[1]):
            #print(j)
            c_diag=np.diag(c[:,j])
            temp_v=np.dot(np.dot(p[:,j],c_diag),u)
            v[j,:]=np.dot(temp_v,np.linalg.pinv(l2_v*np.identity(l)+np.dot(np.dot(u.T,c_diag),u)))
        print('v complete')
        print(np.linalg.norm(p-np.dot(u,v.T)))
        
        W1,b1,c1 = auto.autoEncoder_mono(ratio_l,ratio_u,batch,W1,xtrain,u,b1,c1,accList,EPOCH_NUM,LEARNING_RATE,denoise = True)
        hidden = auto.getoutPut_mono(W1,b1,xtrain,accList)
        u=hidden
        print(np.linalg.norm(p-np.dot(u,v.T)))
    
    with h5py.File('u_40_mono_40+100_auto.h5', 'w') as hf:
        hf.create_dataset("u",  data=u)
    with h5py.File('v_40_mono_40+100_auto.h5', 'w') as hf:
        hf.create_dataset("v",  data=v)
    with h5py.File('W1_40_mono_40+100.h5', 'w') as hf:
        hf.create_dataset("W1",  data=W1)
    with h5py.File('b1_40_mono_40+100.h5', 'w') as hf:
        hf.create_dataset("b1",  data=b1)
    with h5py.File('c1_40_mono_40+100.h5', 'w') as hf:
        hf.create_dataset("c1",  data=c1)
    
    #making_graph(trainLoss, valiLoss, testLoss)
    
    #
    # hidden = pd.DataFrame(hidden.T)
    # prediction = pd.DataFrame(prediction.T)
    # hidden.to_csv('hidden.csv')
    # prediction.to_csv('prediction.csv')
    
    return hidden

main(denoise = True)
