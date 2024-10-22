# This documentation is currently incomplete.

# NeuralNetwork
A Python tool to build neural networks. 

## Usage
`from NeuralNetwork import pynn` 
### pynn.activations
`pynn.activations.step()` is a classic step activation function.

`pynn.activations.relu()` is a ReLU activation function.

`pynn.activations.softmax()` is a softmax activation function, to combine with `pynn.loss_cce()`

### NN = pynn.Neural_Network()
`NN.create(layers, activations)`

- `layers` : tuple containing the sizes of layers--eg `(3,4,4,3)`
- `activations` : tuple containing either instances of the activation functions, or tuple of integers representing the id of each function.

Assigns each created layer to the `NN.layers` list & initialises many instance variables. Call this method before any other.

