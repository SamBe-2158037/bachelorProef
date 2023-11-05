import math
import random
from matplotlib import pyplot as plt
import numpy as npy
PI = 3.14159265358979323846
LEARNINGRATE = 0.01

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
    if(x<-10000):
        return 0
    if(x>10000):
        return 1
    return  1/(1+npy.exp(-1*x))
def sigmoid_vector(arr):
    sigmoid_func = npy.vectorize(sigmoid)
    return sigmoid_func(arr)

#afgeleide activatiefunctie,werkt
def sigmoidDerivative(x):
    y = sigmoid(x)
    return y * (1 - y)


def logit(x):#inverse sigmoid
    if(x==1):
        return 10000
    if(x==0):
        return -10000
    return npy.log(x/(1-x))

#forward propegation: berekent de waardes van de nodes in de volgende layer,werkt
def forwardPropLayer(incomingLayer,outgoingLayer):
    outgoingLayer.nodeValues = npy.dot(outgoingLayer.weights, incomingLayer.nodeValues) + outgoingLayer.bias
    if (outgoingLayer.outgoingNodes != 1):#bij de output line mag sigmoid niet toegepast worden, aangzien output als enigste 1 outgoing node heeft is dit een "goede" tijdelijke oplossing
            outgoingLayer.nodeValues = sigmoid_vector(outgoingLayer.nodeValues)
    return outgoingLayer


def forwardProp(network):#voert forward prop uit op een netwerk(lijst van lagen)
    
    for i in range(1,len(network)):
        network[i] = forwardPropLayer(network[i-1],network[i])
    return network


#bereken de costfunctie: voer forwardpropegation uit met de huidige weights & biases, bereken dan MSE,werkt
def cost(network,TrainingData, size):
    cost = 0
    npy.random.shuffle(TrainingData)
    
    for i in range(size):#TrainingData.shape[0]):
        TrainingDataComponent = TrainingData[i]
        network[0].nodeValues = npy.array([[TrainingDataComponent[0]]])
        network = forwardProp(network)
        cost += (TrainingDataComponent[1]-network[-1].nodeValues[0][0])**2

    cost = cost/size#TrainingData.shape[0]
    return cost

#1ste versie gradient descent: voor elke laag in het netwerk: bereken gradient numeriek met def afgeleide (f(x+h)-f(x))/h)

def gradientDescent(trainingsdata,network):    
    for i in range(1,len(network)):
        gradientWeights = npy.zeros((network[i].outgoingNodes, network[i].incomingNodes))
        gradientBias = npy.zeros((network[i].outgoingNodes, 1))

        #bewerking is hier analoog aan bias maar dan in 2 dimensies
        for k in range(network[i].outgoingNodes):
            gradientBias[k][0] = backprop(trainingsdata,network, i, k,0,bias=True) #voeg def afgeleide toe aan originalCost

            for l in range(network[i].incomingNodes):
                gradientWeights[k][l]= backprop(trainingsdata,network, i, k,l,bias=False)
        #print(gradientWeights)
        #print(gradientBias)
        network[i].bias -= LEARNINGRATE * gradientBias
        network[i].weights -= LEARNINGRATE * gradientWeights    
    return network


def backprop(trainingData, network,layerID, k,l,bias=False):#from l to k
    prevLayer = network[layerID-1]
    currLayer = network[layerID]
    if(~bias):
        a = prevLayer.nodeValues[l][0] # dz_j/dw_jk z_j = sum(w_jk * a_k)+b_j  ==> a=a_k
    else:
        a=1#dz_j/db_j is 1 want staat geen term bij de bias; z_j = sum(w_jk * a_k)+b_j
            
    sigDer = sigmoidDerivative(logit(currLayer.nodeValues[k][0]))#z_j is zie hierboven, aka zonder sigmoid, logit==inverse
    der=0
    for i in range(trainingData.shape[0]):#trainingsize
        der+=2*(currLayer.nodeValues[k][0]-trainingData[i][0])
    return a*sigDer/trainingData.shape[0]*der
        
        


#voor elk datapunt, forwardProp hiermee en voer dan gradientdescent uit
def Learn(network,trainingsdata):
    for k in range(10):
        print("learn routine "+str(k))
        network = gradientDescent(trainingsdata,network)

def main():
    size = 10 #de grootte van de lagen in het midden
    trainingsize = 100

    x=npy.random.uniform(-1*PI,PI, trainingsize)
    trainingsdata = npy.column_stack((x,npy.sin(x)))
    print(trainingsdata)
    layersize = [1, size, size, size]
    network=[layer(0,0,0,1,[[0]])]
    for i in range(1,len(layersize)):# aantal layers
        randweights = npy.array([[random.random() for i in range(network[i-1].outgoingNodes)] for _ in range(layersize[i])])
        randbias = npy.array([[random.random()] for _ in range(layersize[i])])
        network.append(layer(randweights, randbias, network[i-1].outgoingNodes, layersize[i], npy.zeros((layersize[i],1))))
    
    Learn(network,trainingsdata)
    print("netwerk getraind")

    xvals = npy.random.uniform(-1*PI,PI, trainingsize)
    yvals = []
    #hier gebruiken we het netwerk om functiewaardes te benaderen: y = netwerk(x)
    for i in range(trainingsize):
        #print(i)
        network[0].nodeValues = npy.array([[xvals[i]]])
        y = round(forwardProp(network)[-1].nodeValues[0][0],5)
        yvals.append(y)
    #print(xvals)
    #print(yvals)
    plt.scatter(xvals,yvals)
    plt.show()

if __name__ =="__main__":
    main()
