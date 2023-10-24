#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define TRAININGSIZE 100
#define LAYERSIZE 20
#define Pi 3.14159265358979323846f

//activatiefunctie
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}
//afgeleide van activatie functie
double sigmoidDerivative(x) {
    double y = sigmoid(x);
    return y * (1 - y);
}
// bereken de cost functie 
double cost(double TrainingData[][2], double outputLayer[]) {
    double sum = 0;
    for (int i = 0; i < TRAININGSIZE; i++) {
        double eps = TrainingData[i][1] - outputLayer[i];
        sum = sum + eps * eps;

    }
    return sum;
}
// bereken a^(l)
void calcLayer(double **weights,int sizeIN, int sizeOUT, double outgoingNodes[], double bias[], double incomingNodes[]) {
    //nodes from incoming 2 outcoming
    for (int j = 0; j < sizeOUT; j++) {
        *(outgoingNodes + j) = 0;
        for (int i = 0; i < sizeIN; i++) {
            *(outgoingNodes + j) += weights[i][j] * incomingNodes[i];
        }
        *(outgoingNodes + j) = *(outgoingNodes + j) + bias[j];
    }
}

void Learnfunction() {

}

void sinusSetup(double trainingset[][2]) {
    for (int i = 0; i < TRAININGSIZE; i++) {
        trainingset[i][0] = (double)rand() / ((double)RAND_MAX) *200-100;//random in [-100, 100]
        trainingset[i][1] = sin(trainingset[i][0]);//sin(rand)
    }
}

void viewArray(double array[][2]) {
    for (int i = 0; i < TRAININGSIZE; i++) {
        printf("%d) x = %lf \t|| sin(x) = %lf\n",i, array[i][0], array[i][1]);
    }
}

int main() {
    double trainingsData[TRAININGSIZE][2];
    sinusSetup(trainingsData); // [Input , Output]
    viewArray(trainingsData);
    double bias[2][TRAININGSIZE]; // voor laag met size m naar laag met size n => n x 1 array
    double weights[2][TRAININGSIZE]; // voor laag met size m naar laag met size n => n x m array
    //elke rij in weights matrix is voor alle vorige nodes naar 1 volgende te gaan w_00 w_10 w_20 ...

    double lastBias;
    double lastWeight;

    float learningRate = 0.25;

    printf("all done, yeeeey");
    return 0;
}
