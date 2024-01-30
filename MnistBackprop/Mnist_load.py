from keras.datasets import mnist
#from matplotlib import pyplot as plt
import numpy as np
import sys
import time
import AI_python

def main():
    start_time = time.time()
    #loading the dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    trainingsdata = [[np.transpose([train_X[i].flatten()])/255,  np.transpose([np.insert(np.zeros(9),train_y[i],1)])] for i in range(len(train_X)) ]
    ValidationData = [[np.transpose([test_X[i].flatten()])/255, np.transpose([np.insert(np.zeros(9),test_y[i],1)])] for i in range(len(test_X)) ]
    
    print(len(trainingsdata))
    print(len(ValidationData))
    netwerk =  AI_python.initialize_network([28**2, 16,16, 10])
    AI_python.Learn(netwerk,trainingsdata, ValidationData, 35,100 )
    correct= 0
    for i in range(len(ValidationData)):
        netwerk[0].nodeValues = np.array(ValidationData[i][0])
        netwerk = AI_python.forwardProp(netwerk)
        if(100<=i<150):
            print("--------- GUESS ------------------------")
            print(np.argmax(netwerk[-1].nodeValues), end="")
            print("   from: ",end="")
            print(np.transpose(netwerk[-1].nodeValues))
            print("-- correct value: ")
            print(np.argmax(ValidationData[i][1]))
        if(np.argmax(ValidationData[i][1]) == np.argmax(netwerk[-1].nodeValues)):
            correct+=1
    print("_________________________________________________")
    print("CORRECT RATIO: %f"% (correct / len(ValidationData)), end="")
    print("    with correct: "+str(correct)+" out of "+str(len(ValidationData)))
    print("--- %s seconds ----- end at: "% (time.time() - start_time), end="")
    print(time.asctime)



if __name__ =="__main__":
    main()



