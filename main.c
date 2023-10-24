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
    for(unsigned int i =0;i<sizeof(outputLayer)/sizeof(double);i++){
        double eps = TrainingData[i]-outputLayer[i];        
        sum = sum + eps*eps;

    }
    return sum;
}
// bereken a^(l)
void calcLayer(double weights[],double outgoingNodes[],double bias[],double incomingNodes[]){

    if (numOutgoingNodes != sizeof(outgoingNodes) / sizeof(double) || numIncomingNodes != sizeof(incomingNodes) / sizeof(double)) {
        printf("Size mismatch error: The provided array sizes do not match the expected sizes.\n");
        return; // Exit the function
    }
    
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
    double trainingsData[TRAININGSIZE][2]; // [Input , Output]
    double bias[2][TRAININGSIZE]; // voor laag met size m naar laag met size n => 1 x n array
    double weights[2][TRAININGSIZE]; // voor laag met size m naar laag met size n => m x n array

    double lastBias;
    double lastWeight;
 
    float learningRate = 0.25;

    double layer[LAYERSIZE];
    printf("all done, yeeeey");
    return 0;
}
