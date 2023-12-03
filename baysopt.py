import numpy as np
from matplotlib import pyplot as plt


def kernel(X,Y):
    #keuze van kernel is Radial base function 
    sum = 0
    if type(X) != 'list' :
        return np.exp(-(X-Y)**2)
    for i in range(len(X)):
        sum = sum+ (X[i]-Y[i])^2
    return np.exp(-1*sum^2)

def K(x):
    # stel K op
    K = [[kernel(x[0],x[0])]]
    for i in range(1,len(x)):
        K.append([kernel(x[0],x[i])])
        print(K)
        K[0].append(kernel(x[0],x[i]))
        print(K)
        for j in range(1,i+1):
            a = kernel(x[i],x[j])
            K[i].append(a)
            print(K)
            if j!= i:
                K[j].append(a)
                print(K)

    return K
def K_star(X,y):
    K_star = []
    for i in range(0,len(X)):
        K_star.append(kernel(X[i],y))
    return K_star

def K_starstar(y):
    return kernel(y,y)

def predict(X,Xvals,Y):
    k = K(X)
    print(k)
    k = np.linalg.inv(K(X))
    datamean = []
    datavar = []
    for i in range(len(Xvals)):
        kstar = K_star(X,Xvals[i])
        kstarstar = K_starstar(Xvals[i])
        meanstar = np.dot(kstar,np.dot(k,Y))
        cov = kstarstar - np.dot(np.transpose(kstar),np.dot(k,kstar))
        datamean.append(meanstar)
        datavar.append(cov)
    return[datamean,cov]

def GP(X,Y,xRange):
    data = predict(X,xRange,Y)
    plt.scatter(xRange,data[0])
    plt.scatter(X,Y)
    plt.show()



    
        

def main():
    Xvals = [-1,0,1,2,3,4,5,6,7]
    Yvals = [4.526,5,4.923,4.319,3.266,1.894,0.363,-1.149,-2.468]
    xRange = np.arange(-1,7,0.1)
    plt.scatter(xRange,np.sin(xRange/5)+5*np.cos(xRange/3))
    GP(Xvals,Yvals,xRange)


if __name__ == "__main__":
    main()
