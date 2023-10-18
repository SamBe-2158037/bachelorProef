#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define TRAININGSIZE 20
#define LAYERSIZE 20
//activatiefunctie
double sigmoid(double x){
    return 1/(1+ exp(-x));
}
//afgeleide van activatie functie
double sigmoidDerivative(x){
    double y = sigmoid(x);
    return y*(1-y);
}
// bereken de cost functie 
double cost(double TrainingData[],double outputLayer[]){
    double sum = 0;
    for(unsigned char i =0;i<sizeof(outputLayer)/sizeof(double);i++){
        double eps = TrainingData[i]-outputLayer[i];        
        sum = sum + eps*eps;

    }
    return sum;
}
// bereken a^(l)
void calcLayer(double weights[],double outgoingNodes[],double bias[],double incomingNodes[]){
    for(int j =0;j<sizeof(outgoingNodes)/sizeof(double);j++){
        *(outgoingNodes+j) = 0;
        for(int i = 0;i<sizeof(incomingNodes)/sizeof(double);i++){
            *(outgoingNodes+j) = *(outgoingNodes+j) + weights[i]*incomingNodes[i];
        }
        *(outgoingNodes+j) = *(outgoingNodes+j)+bias[j];
    }
}

void Learnfunction(){
    
}


    
int main(){
    double trainingsData[TRAININGSIZE][2];
    double bias[2][TRAININGSIZE];
    double weights[2][TRAININGSIZE];

    double lastBias;
    double lastWeight;
 
    float learningRate = 0.25;

    double layer[LAYERSIZE];

}
