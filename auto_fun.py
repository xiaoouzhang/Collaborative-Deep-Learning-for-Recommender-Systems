import numpy as np
import math

randomSeed = np.random.RandomState(42)

def initialization(INPUT_LAYER,HIDDEN_UNIT,mu,sigma):
    NUM_HIDDEN=len(HIDDEN_UNIT)
    
    W1 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT[0],INPUT_LAYER])
    b1 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT[0]])
    c1 = randomSeed.normal(mu, sigma,[INPUT_LAYER])
    if NUM_HIDDEN==1:
        return W1, b1, c1
    elif NUM_HIDDEN==2:
        W2 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT[1],HIDDEN_UNIT[0]])
        b2 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT[1]])
        c2 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT[0]])
        return W1,W2,b1,b2,c1,c2
    elif HIDDEN==3:
        W2 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT[1],HIDDEN_UNIT[0]])
        b2 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT[1]])
        c2 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT[0]])
        W3 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT[2],HIDDEN_UNIT[1]])
        b3 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT[2]])
        c3 = randomSeed.normal(mu, sigma,[HIDDEN_UNIT[1]])
        
        return W1,W2,W3,b1,b2,b3,c1,c2,c3
    else:
        print('too much layers')
        return 0



def preActivation(W,x,b):
    #a = W.dot(x.T)+ bias
    b_shaped=np.reshape(b,(1,b.size))
    b_shaped=np.repeat(b_shaped,x.shape[0],axis=0)
    y=np.dot(x,W.T)+b_shaped
    return y

def sigmoid_forward(x):
    return 1. / (1 + np.exp(-x))

def sigmoid_backward(x):
    return np.multiply(sigmoid_forward(x),(1-sigmoid_forward(x)))

def tanh_forward(z):
    return np.tanh(z)

def tanh_backward(z):
    return 1-np.square(tanh_forward(z))


def softmax(z):
    #p = np.exp(z)
    #prob = p/p.sum(axis=0)

    z_exp=np.exp(z)
    z_sum=np.sum(z_exp,axis=1)
    z_sum=np.reshape(z_sum,(z_sum.size,1))
    z_sum=np.repeat(z_sum,z.shape[1],axis=1)
    return z_exp/z_sum

def getLoss(W1,W2,xtrain,u,b1,b2,c1,c2,accList):
    loss = 0
    for t in range(1):
        #x = xtrain[t].reshape((1, xtrain.shape[1]))
        #nonZIndex = np.nonzero(x)[1]
        x=xtrain

        A1 = preActivation(W1,x,b1) # batch*hidden
        h1 = tanh_forward(A1)
                
        A2 = preActivation(W2,h1,b2)
        h2 = tanh_forward(A2)
        #h2=A2
        
        A3 = preActivation(W2.T,h2,c2)
        h3 = tanh_forward(A3)
        #h3=h1

        Ahat = preActivation(W1.T,h3,c1) # batch*encode
        
        xhat0 = softmax(Ahat[:,0:accList[0]])  # 784*1
        xhat1 = softmax(Ahat[:,accList[0]:accList[1]])
        xhat2 = softmax(Ahat[:,accList[1]:accList[2]])
        xhat3 = softmax(Ahat[:,accList[2]:accList[3]])
        xhat4 = softmax(Ahat[:,accList[3]:accList[4]])
        xhat5 = softmax(Ahat[:,accList[4]:accList[5]])
        xhat6 = softmax(Ahat[:,accList[5]:accList[6]])
        xhat7 = softmax(Ahat[:,accList[6]:accList[7]])
        xhat8 = softmax(Ahat[:,accList[7]:accList[8]])
        xhat9 = softmax(Ahat[:,accList[8]:accList[9]])
        xhat10 = softmax(Ahat[:,accList[9]:accList[10]])
        xhat11 = softmax(Ahat[:,accList[10]:accList[11]])
        xhat12 = softmax(Ahat[:,accList[11]:accList[12]])
        xhat13 = softmax(Ahat[:,accList[12]:accList[13]])
        xhat14 = softmax(Ahat[:,accList[13]:accList[14]])
        xhat15 = softmax(Ahat[:,accList[14]:accList[15]])
        xhat16 = softmax(Ahat[:,accList[15]:accList[16]])
        xhat17 = softmax(Ahat[:,accList[16]:accList[17]])
        prediction = np.concatenate((xhat0,xhat1,xhat2,xhat3,xhat4,xhat5,xhat6,xhat7,xhat8,xhat9,xhat10, \
                                     xhat11,xhat12,xhat13,xhat14,xhat15,xhat16,xhat17),axis = 1)
        ##calculate loss
        #loss -= np.sum(np.log(prediction[:,nonZIndex]))
        loss -= np.sum(x*np.log(prediction))
    meanLoss = loss/xtrain.shape[0]
    loss_enc=np.sum(np.square(h2-u))/xtrain.shape[0]
    return meanLoss,loss_enc

def autoEncoder(ratio_l,ratio_u,batch,W1,W2,xtrain,u,b1,b2,c1,c2,accList,EPOCH_NUM,LEARNING_RATE,l1,denoise = True):
    ## forward Propogation
    # var0, var1, var2, var3, var4, var5, var6, var7, var8, var9, \
    # var10, var11, var12, var13, var14, var15, var16, var17 = getIndVar(xtrain, lenList)
    beta=0.9
    dW10=np.zeros(W1.shape)
    dW20=np.zeros(W2.shape)
    db10=np.zeros(b1.shape)
    db20=np.zeros(b2.shape)
    dc10=np.zeros(c1.shape)
    dc20=np.zeros(c2.shape)
    for i in range(EPOCH_NUM+1):
        loss = 0
        if i == 0:
            [loss,loss_enc] = getLoss(W1,W2,xtrain,u,b1,b2,c1,c2,accList)
            print (i,loss,loss_enc)
        else:
            for t in range(int(math.floor(xtrain.shape[0]/batch))):
            #for t in range(xtrain.shape[0]):
                x_sample=xtrain[t*batch:(t+1)*batch,:]
                u_sample=u[t*batch:(t+1)*batch,:]
                #x = xtrain[t].reshape((1,xtrain.shape[1]))
                onehot = x_sample
                #nonZIndex = np.nonzero(x)[1]
                if denoise == False:
                    x = x_sample
                else:
                    x = x_sample.astype(float)
                    x += randomSeed.normal(0, 0.1, size= x.shape)

                A1 = preActivation(W1,x,b1) # batch*hidden
                h1 = tanh_forward(A1)
                
                A2 = preActivation(W2,h1,b2)
                h2 = tanh_forward(A2)
                #h2 = (A2)
                
                A3 = preActivation(W2.T,h2,c2)
                h3 = tanh_forward(A3)
                #h3=h1

                Ahat = preActivation(W1.T,h3,c1) # batch*encode
                xhat0 = softmax(Ahat[:,0:accList[0]])  # 784*1
                xhat1 = softmax(Ahat[:,accList[0]:accList[1]])
                xhat2 = softmax(Ahat[:,accList[1]:accList[2]])
                xhat3 = softmax(Ahat[:,accList[2]:accList[3]])
                xhat4 = softmax(Ahat[:,accList[3]:accList[4]])
                xhat5 = softmax(Ahat[:,accList[4]:accList[5]])
                xhat6 = softmax(Ahat[:,accList[5]:accList[6]])
                xhat7 = softmax(Ahat[:,accList[6]:accList[7]])
                xhat8 = softmax(Ahat[:,accList[7]:accList[8]])
                xhat9 = softmax(Ahat[:,accList[8]:accList[9]])
                xhat10 = softmax(Ahat[:,accList[9]:accList[10]])
                xhat11 = softmax(Ahat[:,accList[10]:accList[11]])
                xhat12 = softmax(Ahat[:,accList[11]:accList[12]])
                xhat13 = softmax(Ahat[:,accList[12]:accList[13]])
                xhat14 = softmax(Ahat[:,accList[13]:accList[14]])
                xhat15 = softmax(Ahat[:,accList[14]:accList[15]])
                xhat16 = softmax(Ahat[:,accList[15]:accList[16]])
                xhat17 = softmax(Ahat[:,accList[16]:accList[17]])
                prediction = np.concatenate((xhat0,xhat1,xhat2,xhat3,xhat4,xhat5,xhat6,xhat7,xhat8,xhat9,xhat10, \
                                             xhat11,xhat12,xhat13,xhat14,xhat15,xhat16,xhat17),axis = 1)
                ##calculate loss
                #loss -= np.sum(np.log(prediction[nonZIndex]))

                dLdAhat = prediction-onehot   # batch*encode
                dLdW1_out = np.dot(dLdAhat.T,h3)  # encode*hidden1
                dLdc1 = np.sum(dLdAhat,axis=0)
                dLdh3 = np.dot(dLdAhat,W1.T)  #batch*hidden1

                dLdA3 = np.multiply(dLdh3,tanh_backward(A3))
                dLdW2_out = np.dot(dLdA3.T,h2)    #hidden1*hidden2
                dLdc2 = np.sum(dLdA3,axis=0)
                dLdh2 = np.dot(dLdA3,W2.T)    #batch*hidden2

                dLdA2 = np.multiply(dLdh2,tanh_backward(A2))
                #dLdA2=dLdh2
                dLdW2_in= np.dot(dLdA2.T,h1) #hidden2*hidden1
                dLdb2 = np.sum(dLdA2,axis=0)
                dLdh1 = np.dot(dLdA2,W2)   #batch*hidden1
                #dLdh1=dLdh3

                dLdA1 = np.multiply(dLdh1,tanh_backward(A1))
                dLdW1_in = np.dot(dLdA1.T,x)
                dLdb1=np.sum(dLdA1,axis=0)
                
                
                dudh2=h2-u_sample
                dudA2 = np.multiply(dudh2,tanh_backward(A2))
                #dudA2=dudh2
                dudW2= np.dot(dudA2.T,h1) #hidden2*hidden1
                dudb2 = np.sum(dudA2,axis=0)
                dudh1 = np.dot(dudA2,W2)   #batch*hidden1

                dudA1 = np.multiply(dudh1,tanh_backward(A1))
                dudW1 = np.dot(dudA1.T,x)
                dudb1 =np.sum(dudA1,axis=0)
                
                # update parameters
                #dudW1=0
                #dudW2=0
                #dudb1=0
                #dudb2=0
                dW1=((dLdW1_out.T + dLdW1_in)/ratio_l+dudW1/ratio_u+l1*np.sign(W1))
                W1 += -LEARNING_RATE * (dW1+beta*dW10)
                dW10=(dW1+beta*dW10)
                dW2=((dLdW2_out.T + dLdW2_in)/ratio_l+dudW2/ratio_u+l1*np.sign(W2))
                W2 += -LEARNING_RATE * (dW2+beta*dW20)
                dW20=(dW2+beta*dW20)
                db1=(dLdb1/ratio_l + dudb1/ratio_u)
                b1 += -LEARNING_RATE * (db1+beta*db10)
                db10=(db1+beta*db10)
                db2=(dLdb2/ratio_l + dudb2/ratio_u)
                b2 += -LEARNING_RATE * (db2+beta*db20)
                db20=(db2+beta*db20)
                dc1=dLdc1/ratio_l
                c1 += -LEARNING_RATE * (dc1+beta*dc10)
                dc10=(dc1+beta*dc10)
                dc2=dLdc2/ratio_l
                c2 += -LEARNING_RATE * (dc2+beta*dc20)
                dc20=(dc2+beta*dc20)

            [loss,loss_enc] = getLoss(W1,W2,xtrain,u,b1,b2,c1,c2,accList)
            print (i,loss,loss_enc)
    return W1,W2,b1,b2,c1,c2

def making_graph(lossTrain,lossVal,lossTest):
    lossTrain = np.array(lossTrain)
    lossVal = np.array(lossVal)
    lossTest = np.array(lossTest)
    maxval = max(np.max(lossTrain),np.max(lossVal),np.max(lossTest))
    epoch = np.array(range(EPOCH_NUM))
    trainline, = plt.plot(epoch,lossTrain,'r',label='training')
    testLine, = plt.plot(epoch,lossVal, 'g',label='test')
    valiLine, = plt.plot(epoch,lossTest, 'b',label='validation')

    plt.legend(handles=[trainline, valiLine, testLine])

    plt.title('Cross Entropy Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim([-0.05, EPOCH_NUM])
    plt.ylim([-0.05, maxval+2])


def getoutPut(W1,W2,b1,b2,x,accList):
    #A = preActivation(W1, x, b1)  # 20*64
    #hx = tanh_forward(A)  # 20*64
    A1 = preActivation(W1,x,b1) # batch*hidden
    h1 = tanh_forward(A1)
                
    A2 = preActivation(W2,h1,b2)
    h2 = tanh_forward(A2)
    #h2=A2
    return h2

def autoEncoder_mono(ratio_l,ratio_u,batch,W1,xtrain,u,b1,c1,accList,EPOCH_NUM,LEARNING_RATE,l1,denoise = True):
    ## forward Propogation
    # var0, var1, var2, var3, var4, var5, var6, var7, var8, var9, \
    # var10, var11, var12, var13, var14, var15, var16, var17 = getIndVar(xtrain, lenList)
    for i in range(EPOCH_NUM+1):
        loss = 0
        if i == 0:
            [loss,loss_enc] = getLoss_mono(W1,xtrain,u,b1,c1,accList)
            print (i,loss,loss_enc)
        else:
            for t in range(int(math.floor(xtrain.shape[0]/batch))):
            #for t in range(xtrain.shape[0]):
                x_sample=xtrain[t*batch:(t+1)*batch,:]
                u_sample=u[t*batch:(t+1)*batch,:]
                #x = xtrain[t].reshape((1,xtrain.shape[1]))
                onehot = x_sample
                #nonZIndex = np.nonzero(x)[1]
                if denoise == False:
                    x = x_sample
                else:
                    x = x_sample.astype(float)
                    x += randomSeed.normal(0, 0.1, size= x.shape)

                A1 = preActivation(W1,x,b1) # batch*hidden
                h1 = tanh_forward(A1)

                Ahat = preActivation(W1.T,h1,c1) # batch*encode
                xhat0 = softmax(Ahat[:,0:accList[0]])  # 784*1
                xhat1 = softmax(Ahat[:,accList[0]:accList[1]])
                xhat2 = softmax(Ahat[:,accList[1]:accList[2]])
                xhat3 = softmax(Ahat[:,accList[2]:accList[3]])
                xhat4 = softmax(Ahat[:,accList[3]:accList[4]])
                xhat5 = softmax(Ahat[:,accList[4]:accList[5]])
                xhat6 = softmax(Ahat[:,accList[5]:accList[6]])
                xhat7 = softmax(Ahat[:,accList[6]:accList[7]])
                xhat8 = softmax(Ahat[:,accList[7]:accList[8]])
                xhat9 = softmax(Ahat[:,accList[8]:accList[9]])
                xhat10 = softmax(Ahat[:,accList[9]:accList[10]])
                xhat11 = softmax(Ahat[:,accList[10]:accList[11]])
                xhat12 = softmax(Ahat[:,accList[11]:accList[12]])
                xhat13 = softmax(Ahat[:,accList[12]:accList[13]])
                xhat14 = softmax(Ahat[:,accList[13]:accList[14]])
                xhat15 = softmax(Ahat[:,accList[14]:accList[15]])
                xhat16 = softmax(Ahat[:,accList[15]:accList[16]])
                xhat17 = softmax(Ahat[:,accList[16]:accList[17]])
                prediction = np.concatenate((xhat0,xhat1,xhat2,xhat3,xhat4,xhat5,xhat6,xhat7,xhat8,xhat9,xhat10, \
                                             xhat11,xhat12,xhat13,xhat14,xhat15,xhat16,xhat17),axis = 1)
                ##calculate loss
                #loss -= np.sum(np.log(prediction[nonZIndex]))

                dLdAhat = prediction-onehot   # batch*encode
                dLdW1_out = np.dot(dLdAhat.T,h1)  # encode*hidden1
                dLdc1 = np.sum(dLdAhat,axis=0)
                dLdh1 = np.dot(dLdAhat,W1.T)  #batch*hidden1
                #dLdh1=dLdh3

                dLdA1 = np.multiply(dLdh1,tanh_backward(A1))
                dLdW1_in = np.dot(dLdA1.T,x)
                dLdb1=np.sum(dLdA1,axis=0)
                
                
                dudh1=h1-u_sample
                dudA1 = np.multiply(dudh1,tanh_backward(A1))
                dudW1 = np.dot(dudA1.T,x)
                dudb1 =np.sum(dudA1,axis=0)
                
                # update parameters
                #dudW1=0
                #dudW2=0
                #dudb1=0
                #dudb2=0
                
                W1 += -LEARNING_RATE * ((dLdW1_out.T + dLdW1_in)/ratio_l+dudW1/ratio_u+l1*np.sign(W1))
                b1 += -LEARNING_RATE * (dLdb1/ratio_l + dudb1/ratio_u)
                c1 += -LEARNING_RATE * dLdc1/ratio_l

            [loss,loss_enc] = getLoss_mono(W1,xtrain,u,b1,c1,accList)
            print (i,loss,loss_enc)
    return W1,b1,c1            
def getLoss_mono(W1,xtrain,u,b1,c1,accList):
    loss = 0
    for t in range(1):
        #x = xtrain[t].reshape((1, xtrain.shape[1]))
        #nonZIndex = np.nonzero(x)[1]
        x=xtrain

        A1 = preActivation(W1,x,b1) # batch*hidden
        h1 = tanh_forward(A1)

        Ahat = preActivation(W1.T,h1,c1) # batch*encode
        
        xhat0 = softmax(Ahat[:,0:accList[0]])  # 784*1
        xhat1 = softmax(Ahat[:,accList[0]:accList[1]])
        xhat2 = softmax(Ahat[:,accList[1]:accList[2]])
        xhat3 = softmax(Ahat[:,accList[2]:accList[3]])
        xhat4 = softmax(Ahat[:,accList[3]:accList[4]])
        xhat5 = softmax(Ahat[:,accList[4]:accList[5]])
        xhat6 = softmax(Ahat[:,accList[5]:accList[6]])
        xhat7 = softmax(Ahat[:,accList[6]:accList[7]])
        xhat8 = softmax(Ahat[:,accList[7]:accList[8]])
        xhat9 = softmax(Ahat[:,accList[8]:accList[9]])
        xhat10 = softmax(Ahat[:,accList[9]:accList[10]])
        xhat11 = softmax(Ahat[:,accList[10]:accList[11]])
        xhat12 = softmax(Ahat[:,accList[11]:accList[12]])
        xhat13 = softmax(Ahat[:,accList[12]:accList[13]])
        xhat14 = softmax(Ahat[:,accList[13]:accList[14]])
        xhat15 = softmax(Ahat[:,accList[14]:accList[15]])
        xhat16 = softmax(Ahat[:,accList[15]:accList[16]])
        xhat17 = softmax(Ahat[:,accList[16]:accList[17]])
        prediction = np.concatenate((xhat0,xhat1,xhat2,xhat3,xhat4,xhat5,xhat6,xhat7,xhat8,xhat9,xhat10, \
                                     xhat11,xhat12,xhat13,xhat14,xhat15,xhat16,xhat17),axis = 1)
        ##calculate loss
        #loss -= np.sum(np.log(prediction[:,nonZIndex]))
        loss -= np.sum(x*np.log(prediction))
    meanLoss = loss/xtrain.shape[0]
    loss_enc=np.sum(np.square(h1-u))/xtrain.shape[0]
    return meanLoss,loss_enc

def getoutPut_mono(W1,b1,x,accList):
    #A = preActivation(W1, x, b1)  # 20*64
    #hx = tanh_forward(A)  # 20*64
    A1 = preActivation(W1,x,b1) # batch*hidden
    h1 = tanh_forward(A1)
    
    return h1

