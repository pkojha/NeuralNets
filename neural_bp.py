__author__ = 'pojha'

# Things to include
# 1. early stopping
# 2. epoch
# 3. other features

import math
import random
import numpy

class Neuron:

    # define bias and weights
    def __init__(self,bias):
        self.bias = bias
        self.weights = []

    # get the outputs
    def get_output(self, inputs):
        self.inputs = inputs
        self.output = self.logistic(self.get_net_total_input())
        return self.output

    # compute the total net input
    def get_net_total_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias


    # compute the logistic on input
    def logistic(self,total_net_input):
        return 1/(1 + math.exp(-total_net_input))


    def get_pd_error_wrt_total_net_input(self,target_output):
        return self.get_pd_error_wrt_output(target_output) * self.get_pd_total_net_input_wrt_input()

    # get error for each neuron (mean square error)
    def get_error(self,target_output):
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivative of the error with respect to actual output

    def get_pd_error_wrt_output(self,target_output):
        return (-1 * (target_output - self.output))

    # The total net input into the neuron is computed using logistic function to calculate the neuron's output

    def get_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:


    def get_pd_total_net_input_wrt_weight(self,index):
        return self.inputs[index]


class NNetLayer:

    def __init__(self, num_neurons, bias):
        # Every neuron in a layer shares the same bias
        if bias:
            self.bias = bias
        else:
            self.bias = random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def printSelf(self):
        print "Neurons" , len(self.neurons)
        for n in range(len(self.neurons)):
            print "neuron" , n
            for wts in range(len(self.neurons[n].weights)):
                print "Weight: " ,self.neurons[n].weights[wts]
            print "Bias: ",self.bias

    def feed_forward(self,inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.get_output(inputs))
        return outputs

    def get_outputs(self,inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class NNet:
    LEARNING_RATE = 0.6

    def __init__(self,num_inputs,num_hidden,num_outputs,hidden_layer_weights = None,
                 hidden_layer_bias= None,output_layer_weights=None,output_layer_bias=None):
        self.num_inputs = num_inputs
        self.hidden_layer = NNetLayer(num_hidden,hidden_layer_bias)
        self.output_layer = NNetLayer(num_outputs,output_layer_bias)
        self.init_weights_from_inputs_to_hidden_layer(hidden_layer_weights)
        self.init_weights_from_hidden_layer_to_output_layer(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_to_output_layer(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def printSelf(self):
        print "------"
        print "* Inputs : {}".format(self.num_inputs)
        print "------"
        print "Hidden Layer"
        self.hidden_layer.printSelf()
        print "------"
        print "* Output Layer"
        self.output_layer.printSelf()
        print "------"

    def feed_forward(self,inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)


    def train(self,training_inputs,training_outputs):
        self.feed_forward(training_inputs)

        # output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)

        for o in range(len(self.output_layer.neurons)):
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].\
                                                                    get_pd_error_wrt_total_net_input(training_outputs[o])


        pd_errors_wrt_hidden_layer_total_net_input = [0] * len(self.hidden_layer.neurons)

        # Hidden neuron deltas
        for h in range(len(self.hidden_layer.neurons)):

            d_error_wrt_hidden_layer_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_layer_output += pd_errors_wrt_output_neuron_total_net_input[o] \
                                                   * self.output_layer.neurons[o].weights[h]

            pd_errors_wrt_hidden_layer_total_net_input[h] = d_error_wrt_hidden_layer_output \
                                                            * self.hidden_layer.neurons[h].get_pd_total_net_input_wrt_input()

        # Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] \
                                      * self.output_layer.neurons[o].get_pd_total_net_input_wrt_weight(w_ho)

                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        #  Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                pd_error_wrt_weight = pd_errors_wrt_hidden_layer_total_net_input[h] \
                                      * self.hidden_layer.neurons[h].get_pd_total_net_input_wrt_weight(w_ih)

                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def get_total_error(self,training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].get_error(training_outputs[o])
        return total_error


    ###
    # example 1 #

    # XOR example:

'''
nn = NNet(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
for i in range(10000):
    nn.train([0.05, 0.1], [0.01, 0.99])
    print(i, round(nn.get_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

'''
training_sets = [[[0, 0], [0]],
                     [[0, 1], [1]],
                     [[1, 0], [1]],
                     [[1, 1], [0]]]

nn = NNet(len(training_sets[0][0]), 5, len(training_sets[0][1]),#hidden_layer_weights=[0.15, 0.2, 0.25, 0.3 , 0.4 , 0.2 , 0.6 , 0.7],
  hidden_layer_bias=0.9, #output_layer_weights=[0.1, 0.4],
    output_layer_bias=0.9 )
for i in range(10000):
    training_inputs, training_outputs = random.choice(training_sets)
    nn.train(training_inputs, training_outputs)
    print i, nn.get_total_error(training_sets)



