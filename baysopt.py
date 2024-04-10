import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

def BLACK_BOX(x):
    return np.sin(x/5)+5*np.cos(x/3)
def BB(x,function):
    bb = np.vectorize(function)
    return bb(x)


def kernel(X,Y):
    return np.exp(-(X-Y)**2)

def predict(X, Xvals, Y, nugget = 0.1):
    k = kernel(X, np.transpose(X))  # Calculate the kernel matrix
    k += np.eye(len(X)) * nugget  # Add nugget to the diagonal
    
    if k.shape == (1, 1):
        kinv = np.array([[1 / k[0, 0]]])
    else:
        kinv = np.linalg.inv(k)
    datamean = []
    datavar = []
    for i in range(len(Xvals)):
        k_ = kernel(X, Xvals[i])
        k__ = kernel(Xvals[i], Xvals[i])

        meanstar = np.dot(np.transpose(k_), np.dot(kinv, Y))[0][0]
        cov = k__ - np.dot(np.transpose(k_), np.dot(kinv, k_))[0][0]
        datamean.append(meanstar)
        datavar.append(cov)

    return [datamean, datavar]




def acquisition(X,Xvals,Y,Data, eps):
    xnew = Xvals[0]
    PImax = 0

    imax = np.argmax(Y)
    Xmax = X[imax]

    for i in range(len(Xvals)):
        mean  = Data[0][i]
        var = Data[1][i]
        if(norm.cdf((mean - Y[imax] - eps)/(var+0.01)) > PImax):
            xnew = Xvals[i]
            PImax = norm.cdf((mean - Y[imax] - eps)/(var+0.01))
    print("new point with maximum improvement at: "+str(xnew))
    return xnew
        
def search(Xvals,function):
    X=np.array([[0.1] ,[0.4], [0.5],[0.3]]) #start point in middle?
    Y=BB(X,function)
    for epoch in range(7):
        Data = predict(X,Xvals,Y)

        plt.clf()

       
        plt.plot(Xvals,Data[0],c="blue",linestyle='-')
        plt.plot(Xvals,np.array(Data[0])+np.array(Data[1]),c="red",linestyle='-')
        plt.plot(Xvals,np.array(Data[0])-np.array(Data[1]),c="red",linestyle='-') 
        plt.show()
        xnew = acquisition(X,Xvals,Y,Data,10)
        X = np.append(X,[xnew])
        X = np.transpose([X])
        Y=BB(X,function)




def main():
    xRange = np.arange(-1,7,0.1)
    search(xRange,BLACK_BOX)
    

if __name__ == "__main__":
    main()
