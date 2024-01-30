import math
import random
from turtle import window_height
#from matplotlib import pyplot as plt
#from matplotlib.colors import LogNorm, Normalize
import numpy as np
LEARNINGRATE =.1

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
    return  1/(1+np.exp(-1*x))
def sigmoid_vector(arr):
    sigmoid_func = np.vectorize(sigmoid)
    return sigmoid_func(arr)

def ReLu(x):
    return  max(x,0)
def ReLu_vector(arr):
    F = np.vectorize(ReLu)
    return F(arr)

def CunTanH(x):
    return 1.7159*np.tanh(x*2/3)
def CunTanH_vector(arr):
    cth = np.vectorize(CunTanH)
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
    outgoingLayer.nodeValues = np.dot(outgoingLayer.weights, incomingLayer.nodeValues) + outgoingLayer.bias
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
    mean= np.transpose([np.zeros(len(TrainingData[0][1]))])
    for i in range(len(TrainingData)):
        mean+=TrainingData[i][1]
    mean = mean/len(TrainingData)

    cost = 0#SSE
    SStot=0

    delC = len(TrainingData[0][1])*[0]
    for i in range(len(TrainingData)):
        TrainingDataComponent = TrainingData[i]
        network[0].nodeValues = np.array(TrainingDataComponent[0])
        network = forwardProp(network)

        cost += np.sum((TrainingDataComponent[1]-network[-1].nodeValues)**2)
        SStot += np.sum((TrainingDataComponent[1] - mean)**2)#int(TrainingDataComponent[1][0]!=np.argmax(network[-1].nodeValues))
        for j in range(len(TrainingData[0][1])):
            delC[j] = delC[j] + (network[-1].nodeValues[j]-TrainingDataComponent[1][j])
        #print(network[-1].nodeValues)
        #print(TrainingDataComponent[1])
        

    Rsq = 1 - cost/SStot
    cost = cost/len(TrainingData)
    for j in range(len(delC)):
        delC[j] = 2*delC[j]/len(TrainingData)
    return [cost, Rsq]

def backprop(datapoints, network):
    num_datapoints = len(datapoints)
    num_outputs = len(datapoints[0][1])

    delC = np.zeros(num_outputs)
    GRADW = [np.zeros((node.outgoingNodes, node.incomingNodes)) for node in network]
    GRADB = [np.zeros((node.outgoingNodes, 1)) for node in network]

    for datapoint in datapoints:
        input_values, target_values = datapoint
        network[0].nodeValues = np.array(input_values)
        network = forwardProp(network)

        delC = 2 * (network[-1].nodeValues - target_values)

        for layer in range(len(network) - 1, 0, -1):
            GRADW[layer] += np.outer(delC * (1 - network[layer].nodeValues) * network[layer].nodeValues, network[layer - 1].nodeValues) / num_datapoints

            GRADB[layer] += (delC * network[layer].nodeValues * (1 - network[layer].nodeValues)).reshape(-1, 1) / num_datapoints

            delC = np.dot(network[layer].weights.T, delC) * (1 - network[layer - 1].nodeValues) * network[layer - 1].nodeValues

    for i in range(len(network)):
        network[i].bias -= LEARNINGRATE * GRADB[i]
        network[i].weights -= LEARNINGRATE * GRADW[i]

    return network



def backpropnetwork(network, trainingsdata, minibatch):
    for i in range(len(trainingsdata)//minibatch):
        datapoints = trainingsdata[i:minibatch+i]
        network = backprop(datapoints, network)

    return network



#voor elk datapunt, forwardProp hiermee en voer dan gradientdescent uit
def Learn(network,trainingsdata, ValidationData, EPOCHS, minibatch):
    moments=[[],[],[],[]]
    ERROR=[]

    check_validation=[0]*5
    for epoch in range(EPOCHS):
        print("learn routine "+str(epoch))
        network = backpropnetwork(network,trainingsdata,minibatch)
        l = cost(network,trainingsdata)[0]
        ERROR.append(l)
        print(l)
        
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
    #plt.plot(list(range(epoch+1)),o - c*np.exp(-k*np.array(range(epoch+1))), linestyle='-')

        


def initialize_network(layersize):
    network=[layer(0,0,0,layersize[0],[[0]*layersize[0]])]
    for i in range(1,len(layersize)):# aantal layers
        randweights = np.array([[random.random()*2-1 for i in range(network[i-1].outgoingNodes)] for _ in range(layersize[i])])
        randbias = np.array([[random.random()*2-1] for _ in range(layersize[i])])
        network.append(layer(randweights, randbias, network[i-1].outgoingNodes, layersize[i], np.zeros((layersize[i],1))))
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
        meanList.append(np.mean(segment))
    return meanList
