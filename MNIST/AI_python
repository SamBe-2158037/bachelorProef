import math
import random
from turtle import window_height
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as npy
import pandas as pd
import seaborn as sns
PI = 3.14159265358979323846
LEARNINGRATE = 0.1

#class van een layer: bevat de weights,biases en nodeswaardes van een rij nodes en de gewichten die binnen komen(de input layer wordt appart gedef aangezien hier niks binnenkomt)
class layer:
    def __init__(self,weights,bias,incomingNodes,outgoingNodes,nodeValues):
        self.weights = weights
        self.bias = bias
        self.incomingNodes = incomingNodes
        self.outgoingNodes = outgoingNodes
        self.nodeValues = nodeValues

#activatie functie, werkt
def sigmoid(x):
    return  1/(1+npy.exp(-1*x))
def sigmoid_vector(arr):
    sigmoid_func = npy.vectorize(sigmoid)
    return sigmoid_func(arr)

def ReLu(x):
    return  max(x,0)
def ReLu_vector(arr):
    F = npy.vectorize(ReLu)
    return F(arr)

def CunTanH(x):
    return 1.7159*npy.tanh(x*2/3)
def CunTanH_vector(arr):
    cth = npy.vectorize(CunTanH)
    return cth(arr)

#afgeleide activatiefunctie,werkt
def sigmoidDerivative(x):
    y = sigmoid(x)
    return y * (1 - y)


#forward propegation: berekent de waardes van de nodes in de volgende layer,werkt
def forwardPropLayer(incomingLayer,outgoingLayer,sigmoidBool):
    #print("\n\n\nINCOMING VALUES")
    #print(incomingLayer.nodeValues)
    #print("\n OUTGOING VALUES")
    #print(outgoingLayer.nodeValues)
    outgoingLayer.nodeValues = npy.dot(outgoingLayer.weights, incomingLayer.nodeValues) + outgoingLayer.bias
    if (sigmoidBool):#bij de output line mag sigmoid niet toegepast worden, aangzien output als enigste 1 outgoing node heeft is dit een "goede" tijdelijke oplossing
            outgoingLayer.nodeValues = sigmoid_vector(outgoingLayer.nodeValues)
    else:
            outgoingLayer.nodeValues = ReLu_vector(outgoingLayer.nodeValues)

    return outgoingLayer


def forwardProp(network):#voert forward prop uit op een netwerk(lijst van lagen)
    
    for i in range(1,len(network)):
        network[i] = forwardPropLayer(network[i-1],network[i],True)
    return network


#bereken de costfunctie: voer forwardpropegation uit met de huidige weights & biases, bereken dan MSE,werkt
def cost(network,TrainingData):
    mean= npy.transpose([npy.zeros(len(TrainingData[0][1]))])
    for i in range(len(TrainingData)):
        mean+=TrainingData[i][1]
    mean = mean/len(TrainingData)

    cost = 0#SSE
    SStot=0
    for i in range(len(TrainingData)):
        TrainingDataComponent = TrainingData[i]
        network[0].nodeValues = npy.array(TrainingDataComponent[0])
        network = forwardProp(network)
        #print(network[-1].nodeValues)
        #print(TrainingDataComponent[1])
        cost += npy.sum((TrainingDataComponent[1]-network[-1].nodeValues)**2)#int(TrainingDataComponent[1][0]!=npy.argmax(network[-1].nodeValues))
        SStot += npy.sum((TrainingDataComponent[1] - mean)**2)
    Rsq = 1 - cost/SStot
    cost = cost/len(TrainingData)
    return [cost, Rsq]

#1ste versie gradient descent: voor elke laag in het netwerk: bereken gradient numeriek met def afgeleide (f(x+h)-f(x))/h)
def ADAM(trainingsdata,network,moments,epoch, adamBool,miniBatch):
    npy.random.shuffle(trainingsdata)
    datapoints = trainingsdata[:miniBatch]
    b1=0.9
    b2=0.999
    eps=10**(-7)

    originalCost = cost(network,datapoints)[0]
    h = 0.1

    GRADbias = []
    GRADweights = []

    for i in range(1,len(network)):
        gradientWeights = npy.zeros((network[i].outgoingNodes, network[i].incomingNodes))
        gradientBias = npy.zeros((network[i].outgoingNodes, 1))

        #bewerking is hier analoog aan bias maar dan in 2 dimensies
        for k in range(network[i].outgoingNodes):
            network[i].bias[k][0] += h #(x+h)
            Cost = cost(network,datapoints)[0]#f(x+h) 
            gradientBias[k][0] = ((Cost-originalCost)/h) #voeg def afgeleide toe aan originalCost
            network[i].bias[k][0] -=h #trek h er terug vanaf zodat deze geen impact heeft op volgende berekeningen

            for l in range(network[i].incomingNodes):
                network[i].weights[k][l] +=h
                Cost = cost(network,datapoints)[0]
                gradientWeights[k][l]=(Cost-originalCost)/h
                network[i].weights[k][l] -=h     
        #print(gradientWeights)
        #print(gradientBias)
        #print(network[i].bias)
        GRADbias.append(gradientBias)
        GRADweights.append(gradientWeights)


    if(adamBool):
        for i in range(len(network)-1):
            if(epoch==0):
                moments[0].append((1-b1)*GRADbias[i])
                moments[1].append((1-b2)*GRADbias[i]**2)
                moments[2].append((1-b1)*GRADweights[i])
                moments[3].append((1-b2)*GRADweights[i]**2)
            else:
                moments[0][i] = b1*moments[0][i]+(1-b1)*GRADbias[i]
                moments[1][i] = b2*moments[1][i]+(1-b2)*GRADbias[i]**2
                moments[2][i] = b1*moments[2][i]+(1-b1)*GRADweights[i]
                moments[3][i] = b2*moments[3][i]+(1-b2)*GRADweights[i]**2
            mbias = moments[0][i]/(1-b1**(epoch+int(epoch==0)))
            vbias = moments[1][i]/(1-b2**(epoch+int(epoch==0)))
            mweight = moments[2][i]/(1-b1**(epoch+int(epoch==0)))
            vweight = moments[3][i]/(1-b2**(epoch+int(epoch==0))) 
        
            network[i+1].bias -= LEARNINGRATE * npy.multiply(mbias,1/(npy.sqrt(vbias)+eps))
            network[i+1].weights -= LEARNINGRATE * npy.multiply(mweight,1/(npy.sqrt(vweight)+eps))
    else:
        for i in range(len(network)-1):
            network[i+1].bias -= LEARNINGRATE * GRADbias[i]
            network[i+1].weights -= LEARNINGRATE * GRADweights[i]
    return network

#voor elk datapunt, forwardProp hiermee en voer dan gradientdescent uit
def Learn(network,trainingsdata, ValidationData, EPOCHS, minibatch):
    moments=[[],[],[],[]]
    ERROR=[]

    check_validation=[0]*100
    for epoch in range(EPOCHS):
        print("learn routine "+str(epoch))
        network = ADAM(trainingsdata,network,moments,epoch,True,minibatch)
        ERROR.append(cost(network,trainingsdata)[0])
        check_validation.append(cost(network,ValidationData)[0])
        check_validation.pop(0)
        a= nearest_neighbor_mean(check_validation,5)
        if(sorted(a)==a and epoch>100   ):
            print("early stop at epoch: "+str(epoch))
            #break


    print("___________BIG TIME ERROR______________")
    print(ERROR)
   # plt.scatter(list(range(epoch+1)), ERROR)
   # plt.show()
    #k=0.08#0.09003271041407755  #0.19703271041407755
    #c=5.221335236601307
    #o=0.9263295899148523
    #plt.plot(list(range(epoch+1)),o - c*npy.exp(-k*npy.array(range(epoch+1))), linestyle='-')

        

def rLoss(k,C,O, Rsqrd):
    summ=0;
    for i in range(len(Rsqrd)):
        summ+=(Rsqrd[i] - O + C*npy.exp(-k*i))**2
    summ = summ/len(Rsqrd)
    return summ

def initialize_network(layersize):
    network=[layer(0,0,0,layersize[0],[[0]*layersize[0]])]
    for i in range(1,len(layersize)):# aantal layers
        randweights = npy.array([[random.random()*2-1 for i in range(network[i-1].outgoingNodes)] for _ in range(layersize[i])])
        randbias = npy.array([[random.random()*2-1] for _ in range(layersize[i])])
        network.append(layer(randweights, randbias, network[i-1].outgoingNodes, layersize[i], npy.zeros((layersize[i],1))))
    return network

def nearest_neighbor_mean(array, windowsize):
    meanList=[]
    for i in range(len(array)):
        if(i<math.floor(windowsize/2)):
            segment = array[0:windowsize]
        elif(i>math.floor(windowsize/2)):
            segment = array[-windowsize:]
        else:
            segment = array[i-math.floor(windowsize/2):i+math.floor(windowsize/2)]
        meanList.append(npy.mean(segment))
    return meanList
