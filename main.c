#include <stdio.h>
#include <stdlib.h>
#include "../libs/nnet.h"


int main(){
	
	neuron *n1 = createNeuron(0.25, SIGMOID, 2, 0.5, -0.5);
	neuron *n2 = createNeuron(-0.25, SIGMOID, 2, -0.5, 0.2);
	neuron *n3 = createNeuron(1, SIGMOID, 2, 1.0, -1);
	neuron *n4 = createNeuron(1, SIGMOID, 3, 0.35, -0.3, 0.75);
	neuron *n5 = createNeuron(-1, SIGMOID, 3, 0.25, -0.4, 0.75);
	neuron *n6 = createNeuron(0.8, SIGMOID, 2, -0.9, 0.9);

	layer l1 = createLayer(3, n1, n2, n4);
	layer l2 = createLayer(2, n4, n5);
	layer l3 = createLayer(1, n6);
	net nt = createNet(3, l1, l2, l3);
	
	double *end = propagate(&nt, 1.0, 1.0);
	printf("%lf\n", end[0]);
	
	return 0;
}
