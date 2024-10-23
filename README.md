# Neural Network Evolutionary Algorithm
This repository contains an implementation of a simple neural network model designed for evolutionary algorithms. It includes various classes and functions for defining and manipulating neural networks, activation functions, loss functions, mutation logic, and population management for evolutionary approaches. Below is a list of all the available classes and functions in the Python file.

# Table of Contents
Classes
activations
loss_func
loss_cce
Neural_Network
generation
Functions
activations.step
activations.relu
activations.softmax
loss_func.calculate
loss_cce.forward
Neural_Network.init
Neural_Network.create
Neural_Network.activate
Neural_Network.update
Neural_Network.save
Neural_Network.mutate
Neural_Network.clone
Neural_Network.neuron_layer.set_weights
Neural_Network.neuron_layer.set_biases
Neural_Network.neuron_layer.mutate
Neural_Network.neuron_layer.forward
generation.init
generation.copy_network
generation.activate_gen
generation.get_best
generation.repopulate
generation.mutate_gen
generation.set_best_wb
generation.save
load

## Classes
### activations
The activations class contains different types of neural network activation functions.

 Inner Classes:
`step`: Step function activation.
`relu`: Rectified Linear Unit (ReLU) activation.
`softmax`: Softmax activation for multi-class classification.

### loss_func
This class defines the base structure for loss functions in the neural network.

Methods:
`calculate`: Calculates the loss value based on the network's output and the target labels.

### loss_cce
Inherits from `loss_func`, implements the categorical cross-entropy (CCE) loss function.

Methods:
`forward`: Computes the forward pass of the categorical cross-entropy loss.

### Neural_Network
This class defines the structure of a neural network and includes methods for mutating, cloning, saving, and updating the model.

Inner Classes:
`neuron_layer`: Represents a single layer in the neural network.
Methods:
`create`: Creates a neural network with the specified layers and activations.
`activate`: Feeds input data through the neural network.
`update`: Updates the neural network's weights and biases if the loss improves.
`save`: Saves the neural network’s weights and biases to a file.
`mutate`: Applies mutations to the neural network for evolutionary algorithms.
`clone`: Creates a deep copy of the current neural network.

### generation
This class handles the evolutionary algorithm by managing a population of neural networks, applying mutations, and selecting the best-performing networks.

Methods:
`copy_network`: Clones the neural network for population creation.
`activate_gen`: Activates the entire generation by feeding input data into each network.
`get_best`: Finds and returns the best neural network in the population.
`repopulate`: Selects and repopulates the generation based on the performance of networks.
`mutate_gen`: Mutates the generation of neural networks.
`set_best_w`b: Sets the best weights and biases for each network.
`save`: Saves the best-performing network in the generation to a file.

## Functions
### activations.step
Represents a step activation function, which outputs 1 if the input is positive, and 0 otherwise.
Methods:
`forward(inputs)`: Performs the forward pass of the step activation function.

### activations.relu
Implements the Rectified Linear Unit (ReLU) activation function.
Methods:
`forward(inputs)`: Performs the forward pass of the ReLU activation.

### activations.softmax
Implements the softmax activation function.
Methods:
`forward(inputs)`: Performs the forward pass of the softmax activation function, outputting probability values for multi-class classification.

### loss_func.calculate(output, y)
Calculates the loss of the network output given the target labels.
Arguments:
`output`: The output from the neural network.
`y`: The true labels.

### loss_cce.forward(y_pred, y_true)
Computes the forward pass of the categorical cross-entropy (CCE) loss function.

Arguments:
`y_pred`: Predicted output from the neural network.
`y_true`: True labels (one-hot or label encoded).

### Neural_Network.init
Initializes the Neural_Network object with basic attributes like iteration and loss.

### Neural_Network.create(sizes, activations)
Creates the neural network structure, setting up the layers and activation functions.

Arguments:
`sizes`: List of integers representing the number of neurons in each layer.
`activations`: List of activation functions for each layer.

### Neural_Network.activate(X)
Feeds input data through the network and computes the output.

Arguments:
`X`: Input data for the network.

### Neural_Network.update(loss,*,verbose=False)
Updates the weights and biases of the network if the new loss is lower than the previous best loss.

Arguments:
`loss`: The loss value of the network.
`verbose`: Whether to print updates.

### Neural_Network.save(file_name)
Saves the neural network's weights and biases to a specified file.

Arguments:
`file_name`: The name of the file to save the neural network to.

### Neural_Network.mutate(limit)
Applies mutation to the neural network by adjusting the weights and biases slightly to explore the solution space.

### Neural_Network.clone()
Creates a deep copy of the neural network.

### Neural_Network.neuron_layer.set_weights(weights)
Manually sets the weights of a specific neuron layer.

Arguments:
`weights`: A numpy array or list representing the new weights.

### Neural_Network.neuron_layer.set_biases(biases)
Manually sets the biases of a specific neuron layer.

Arguments:
`biases`: A numpy array or list representing the new biases.

### Neural_Network.neuron_layer.mutate(limit)
Applies mutation to the weights and biases of a neuron layer.

Arguments:
`limit`: An integer value that limits the mutation range.

### Neural_Network.neuron_layer.forward(inputs)
Performs the forward pass of a neuron layer by calculating the dot product of the inputs and weights and adding biases.

### generation.init(network, size)
Initializes the generation object by creating a population of neural networks.

Arguments:
`network`: The neural network template.
`size`: The size of the population.

### generation.copy_network()
Creates deep copies of a neural network to initialize the population.

### generation.activate_gen(X)
Activates each network in the generation using the input data.

Arguments:
`X`: Input data to be passed through each network.

### generation.get_best(*, get_loss=False)
Sorts the networks based on their performance and returns the best-performing networks.

### generation.repopulate()
Selects and repopulates the generation by copying the best-performing networks and discarding the worst-performing ones.

### generation.mutate_gen(limit)
Applies mutation to the entire generation except for the best-performing network.

Arguments:
`limit`: Mutation range limit.

### generation.set_best_wb()
Sets the best weights and biases for each network in the population.

### generation.save(file_name)
Saves the best network in the generation to a specified file.

Arguments:
`file_name`: The name of the file to save the network.

### load(file_name)
Loads a previously saved generation from a file and returns the corresponding generation object.

Arguments:
`file_name`: The file from which to load the network data.
================================