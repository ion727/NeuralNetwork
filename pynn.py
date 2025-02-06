import numpy as np
from random import random, randint
from copy import deepcopy

class activations:
    class step:
        def __init__(self):
            self.id = 0
        def forward(self, inputs):
            vstep = np.vectorize(lambda x: 1 if x > 0 else 0)
            self.output = vstep(inputs)
            return self.output
    class relu:
        def __init__(self):
            self.id = 1
        def forward(self, inputs):
            self.output = np.maximum(0,inputs)
    class softmax:
        def __init__(self):
            self.id = 2
        def forward(self, inputs):
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            self.output = probs
    activations = [step, relu, softmax] # update when new activation is added
    
class loss_func:
    def calculate(self, output, y):
        sample_losses = self.forward(output,y)  
        data_loss = np.mean(sample_losses)
        self.loss = float(data_loss)
        return float(data_loss)
class loss_cce(loss_func):
    def forward(self, y_pred:np.ndarray, y_true:np.ndarray):
        if type(y_pred) == list:
            y_pred = np.array(y_pred)
        if type(y_true) == list:
            y_true = np.array(y_true)
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if not len(y_pred.shape) in (1,2):
            raise ValueError(f"Incorrect shape for output y: expected 1 or 2 dimentions, got {len(y_pred.shape)}")
        if not len(y_true.shape) in (1,2):
            raise ValueError(f"Incorrect shape for target y: expected 1 or 2 dimentions, got {len(y_true.shape)}")
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_probs = -np.log(correct_confidences)
        return negative_log_probs 

class Neural_Network:
    def __init__(self):
        self.iteration = 0
        self.loss = 999999.0
        self.set_lowest_loss(self.loss)
    
    class neuron_layer:
        def __init__(self, n_inputs, n_neurons, activation) -> None:
            self.n_inputs = n_inputs
            self.n_neurons = n_neurons
            self.weights = np.zeros((n_inputs, n_neurons))
            self.biases = np.zeros((1, n_neurons))
            self.best_weights = self.weights
            self.best_biases  = self.biases
            self.activation = activation
            
        
        def set_weights(self,weights) -> None: # Manually set w/b
            if type(weights) != np.ndarray:
                weights = np.array(weights)
            if len(weights.shape) == 1:
                weights = weights.reshape(1,weights.shape[0]) # Flexibility for any sort of list to be passed without errors.
            if weights.shape != self.weights.shape:
                raise ValueError(f"set_weights expected array of shape {self.weights.shape}, got {weights.shape}")
            self.weights = weights
            self.best_weights = weights
        
        def set_biases(self,biases) -> None: # Manually set w/b
            if type(biases) != np.ndarray:
                biases = np.array(biases)
            if len(biases.shape) == 1:
                biases = biases.reshape(1,biases.shape[0])
            if biases.shape != self.biases.shape:
                raise ValueError(f"set_biases expected array of shape {self.biases.shape}, got {biases.shape}")
            self.biases = biases
            self.best_biases = biases
        
        def mutate(self,limit=None):
            nolimit = True if limit is None else False
            if type(limit) is not int and nolimit is False:
                raise ValueError(f"Neural_Network.neuron_layer.mutate: \"limit\" argument expected int, got {type(limit)}")
            self.weights += 0.10*np.random.randn(self.n_inputs, self.n_neurons)
            self.biases += 0.10*np.random.randn(1, self.n_neurons)
            
            # Randomly make big changes to a layer.
            # Create tables of 0 or 1 and change the chosen w/b drastically
            if randint(0,9) == 0:
                weights_extra = np.random.randint(0,2,(self.n_inputs, self.n_neurons))
                biases_extra = np.random.randint(0,2,(1, self.n_neurons))
                if limit is None:
                    self.weights *= weights_extra
                    self.biases *= biases_extra
                else:
                    self.weights *= weights_extra*limit
                    self.biases *= biases_extra*limit
                    vcap = np.vectorize(lambda x: max(-limit,min(x,limit)))  # clamp down on the values, max and min being limit and -limit
                    self.weights = vcap(self.weights)
                    self.biases = vcap(self.biases)
            return self
        
        def forward(self, inputs) -> None:
            self.activation_input = np.dot(inputs, self.weights) + self.biases
            self.activate(self.activation_input)
        
        def activate(self,input) -> None:
            self.activation.forward(inputs=input)
            self.output = self.activation.output

    def set_lowest_loss(self,lowest_loss) -> None:
        self.lowest_loss = lowest_loss

    def create(self,*args):  # Creates self.layers
        if len(args) != 2:
            raise ValueError(f"Neural_Networks.create((layers), (activations)) expected 2 positional arguments, got {len(args)}")
        self.sizes, self.activations = args
        
        # Flexibility so that a list of ints can be entered
        if type(self.activations) in (list,tuple) and all(type(value) == int for value in self.activations):
            self.activations = [activations.activations[i]() for i in self.activations]
        
        # Validation
        # Ensuring logical amounts have been entered
        for num in self.sizes:
            if type(num) != int:
                raise ValueError(f"function create() expected int, got {type(num)}.")
            if num < 1:
                raise ValueError("Layer size cannot be less than 1.")
        
        
        self.layers = []
        for i in range(len(self.sizes)-2):        # minus 1 bc input layer doesnt require an object
            layer = self.neuron_layer(self.sizes[i], self.sizes[i+1],self.activations[i])
            self.layers.append(layer)
        self.layers.append(self.neuron_layer(self.sizes[-2], self.sizes[-1],self.activations[-1])) # create a list w all the layers
        for layer in self.layers: # prepare to evolve
            layer.best_weights = layer.weights.copy()
            layer.best_biases  = layer.biases.copy()
        return self
    
    def activate(self,X): #Pass X through the neural network
        if type(X) != np.ndarray:
            X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1,X.shape[0]) # Flexibility for any sort of list to be passed without errors.
        elif len(X.shape) != 2:
            raise ValueError(f"Expected input with 1 or 2 dimensions, got {len(X.shape)} dimensions.")
        if X.shape[1] != self.sizes[0]:
            raise ValueError(f"Target inputs have a different shape from NN inputs: expected {self.sizes[0]}, got {X.shape}") # validating X
        
        mailman = X # it's called the mailman because it connects the layers
        for layer in self.layers:
            layer.forward(mailman)
            mailman = layer.output
        self.best_node = np.argmax(mailman)
        self.confidence = round(max(mailman[0]),2)
        return mailman # returns the output
    
    def update(self,loss,*,verbose=False): # (use after mutation) checks if new loss is lower and assigns to best_wb if so. Otherwise, reverts back wb
        if loss is None:
            loss = self.loss
        if not (type(loss) in (float,int)):
            try:
                loss = float(loss)
            except:
                raise ValueError(f"Expected int/float 'loss' argument, got {type(loss)}")
        self.iteration += 1
        if loss <= self.lowest_loss:
            for layer in self.layers: # update all w/b
                layer.best_weights = layer.weights.copy()
                layer.best_biases  = layer.biases.copy()
            self.set_lowest_loss(loss)
            if verbose is True:
                print(f"new set of weights found, iteration:{self.iteration} loss:{loss}")
        else:
            for layer in self.layers: # revert all w/b
                layer.weights = layer.best_weights.copy()
                layer.biases  = layer.best_biases.copy()
        return self
    
    def save(self,file_name,*,gen_size=1,loss=None): # Save wb to file_name
        if loss is None:
            loss = self.lowest_loss
        with open(file_name,"w") as sf:
            activation_ids = "".join(str(layer.activation.id) for layer in self.layers)
            sf.write(f"{gen_size}\n{loss}\n{activation_ids}\n")
            
            for layer in self.layers:
                weights_str = np.array2string(layer.best_weights, separator=', ', max_line_width=np.inf) 
                biases_str = np.array2string(layer.best_biases, separator=', ', max_line_width=np.inf) 
                sf.write(f"{weights_str}\n\n{biases_str}\n\n")    

    def mutate(self,limit):
        for layer in self.layers:
            layer.mutate(limit)
        return self
    
    def clone(self):
        return deepcopy(self)


        

class generation():
    """_summary_
    Creates a population of neural networks, useful for evolutionary algorithms.
    """
    def __init__(self,network:Neural_Network,size:int):
        self.network = network
        self.generation_lowest_loss = network.lowest_loss
        self.best_generation_weights = [layer.best_weights for layer in network.layers]
        self.best_generation_biases = [layer.best_biases for layer in network.layers]
        self.size = size
        self.copy_network(network, size)

    def copy_network(self,network, size):
        self.networks = [network.clone() for _ in range(size)] # creates a list with deep copies of the neural network
        return self
    
    def activate_gen(self,X):
        self.outputs = [network.activate(X) for network in self.networks]
        return self.outputs
    
    def get_best(self,*, get_loss=False):  # Returns (best_gen_w, best_gen_b) by default and if specified the best network and its loss.
        networks = deepcopy(self.networks)
        networks.sort(key=lambda x: x.loss)
        if get_loss is True:
            return networks[0].loss
        else:
            self.generation_lowest_loss = min(networks[0].loss, self.generation_lowest_loss)
            return networks
    
    def repopulate(self,networks=None):
        if type(networks) is None:
            networks = self.networks
        if type(networks) is list and all(type(i) is Neural_Network for i in networks):
            networks.sort(key=lambda x: x.loss)
            total_loss = sum(network.loss for network in networks)
            probabilities = [network.loss/total_loss for network in networks]
            probabilities[-1] = 1
            best_networks = deepcopy(networks)
            n = min(5,len(networks)-1)
            for i in range(n,len(networks)): # (n, x) so that the main n networks are preserved
                cumulative_prob = 0.0
                selected = random()
                for k,prob in enumerate(probabilities):
                    cumulative_prob += prob
                    if selected <= cumulative_prob:
                        best_networks[i] = deepcopy(networks[k])
                        break
            self.networks = best_networks
            del best_networks
        else: 
            raise ValueError(f'generation.repopulate: "networks" argument expected list[int], got {type(networks)} of:\n{[type(i) for i in networks]}')
            
    
    def mutate_gen(self,limit):
        if len(self.networks) == 1:
            for layer in self.networks[0]:
                layer.mutate(limit)
        else:
            for network in self.networks[:]:
                for layer in network.layers:
                    layer.mutate()
        return self.networks

    def set_best_wb(self):
        for network in self.networks:
            for layer in network.layers:
                layer.best_weights = layer.weights.copy()
                layer.best_biases = layer.biases.copy()
        
    def save(self,file_name):
        self.networks[0].save(file_name,gen_size=self.size,loss=self.generation_lowest_loss)
    
def load(file_name) -> generation:
    with open(file_name, "r") as sf:
        lines = sf.readlines() # remove the two trailing whitespace lines
        size = int(lines[0].strip())
        loss = float(lines[1].strip())
        activation_ids = list(map(int, lines[2].strip()))
        weight_bias_data = lines[3:]
    raw_data = ''.join(weight_bias_data).split('\n\n')[:-1]
    arrays = []
    for raw in raw_data:
        arrays.append(np.array(eval(raw)))
    num_layers = len(arrays) // 2
    weights = arrays[::2]
    biases = arrays[1::2]
    sizes = [weights[0].shape[0]] + [weight.shape[1] for weight in weights] 
    NN = Neural_Network()
    NN.create(sizes, activation_ids)
    NN.loss = loss
    NN.set_lowest_loss(loss)
    for i in range(num_layers):
        NN.layers[i].set_weights(weights[i])
        NN.layers[i].set_biases(biases[i])
    gen = generation(NN, size)

    return gen
