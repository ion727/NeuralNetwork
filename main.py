
import numpy as np
from random import randint
from copy import copy

class activations:        
    class step:
        id = 0
        def forward(self, inputs):
            vstep = np.vectorize(lambda x: 1 if x > 0 else 0)
            self.output = vstep(inputs)
            return self.output
    class relu:
        id = 1
        def forward(self, inputs):
            self.output = np.maximum(0,inputs)
    class softmax:
        id = 2
        def forward(self, inputs):
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            self.output = probs
    activations = [step, relu, softmax] # update when new activation is added
    
class loss_func:
    def calculate(self, output, y):
        sample_losses = self.forward(output,y)  # type: ignore
        data_loss = np.mean(sample_losses)
        self.loss = float(data_loss)
        return float(data_loss)
class loss_cce(loss_func):
    def forward(self, y_pred:np.ndarray, y_true:np.ndarray):
        if type(y_pred) == list:
            y_pred = np.array(y_pred)
        if type(y_true) == list:
            y_pred = np.array(y_true)
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
    def __init__(self, *, interrupt=None):
        self.loss = 999999
        self.interrupt = interrupt
        self.set_lowest_loss(self.loss)
        self.iteration = 0
    
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
        
        def mutate(self) -> None:
            self.weights += 0.10*np.random.randn(self.n_inputs, self.n_neurons)
            self.biases += 0.10*np.random.randn(1, self.n_neurons)
            
            # Randomly make big changes to a layer.
            # Create tables of 0 or 1 and change the chosen w/b drastically
            if randint(0,9) == 0:
                weights_extra = np.random.randint(0,2,(self.n_inputs, self.n_neurons))
                biases_extra = np.random.randint(0,2,(1, self.n_neurons))
                # increase change to selected neurons
                weights_extra *= np.random.randint(-3,3,(self.n_inputs, self.n_neurons)) 
                biases_extra  *= np.random.randint(-3,3,(1, self.n_neurons)) 
                self.weights += weights_extra
                self.biases += biases_extra     
        
        def forward(self, inputs) -> None:
            self.activation_input = np.dot(inputs, self.weights) + self.biases
            self.activate(self.activation_input)
        
        def activate(self,input) -> None:
            self.activation.forward(input)
            self.output = self.activation.output

    def set_lowest_loss(self,lowest_loss) -> None:
        self.lowest_loss = lowest_loss

    def create(self,*args):  # Creates self.layers
        if len(args) != 2:
            raise ValueError(f"Neural_Networks.create((layers), (activations)) expected 2 positional arguments, got {len(args)}")
        self.sizes, self.activations = args
        
        # Flexibility so that a list of ints can be entered
        if type(self.activations == list) and all(type(value) == int for value in self.activations):
            self.activations = [activations.activations[i] for i in self.activations]
        
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
        if X.shape[1] != self.sizes[0]:
            raise ValueError(f"Target inputs have a different shape from NN inputs: expected {num_inputs}, got {X.shape}") # validating X
        
        mailman = X # it's called the mailman because it connects the layers
        for layer in self.layers:
            layer.forward(mailman)
            mailman = layer.output
        self.best_node = np.argmax(mailman)
        self.confidence = round(max(mailman[0]),2)
        return mailman # returns the output
    
    def update(self,*,loss=None,verbose=False): # (use after mutation) checks if new loss is lower and assigns to best_wb if so. Otherwise, reverts back wb
        if loss is None:
            loss = self.loss
        if not (type(loss) in (float,int)):
            raise ValueError(f"Expected int/float 'loss' argument, got {type(loss)}")
        self.loss = loss
        self.iteration += 1
        if loss < self.lowest_loss:
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
        self.update()
        if loss is None:
                loss = self.lowest_loss
        with open(file_name,"w") as sf:
            data = []
            activation_ids = ""
            for layer in self.layers:
                data.extend((str(layer.best_weights),"\n",str(layer.best_biases),"\n"))
                activation_ids += str(layer.activation.id)
            data = [str(gen_size),"\n",str(loss),"\n",activation_ids] + data[:-1]
            sf.writelines(data)    

    def mutate(self):
        for layer in self.layers:
            layer.mutate()
        return self
    
    def clone(self):
        return copy(self)


        

class generation:
    def __init__(self,network,size,*,loss_function=None):
        self.network = network
        self.best_generation_weights = [layer.best_weights for layer in network.layers]
        self.best_generation_biases = [layer.best_biases for layer in network.layers]
        self.loss_function = loss_function
        self.size = size
        self.create_gen(network, size)

    def create_gen(self,network, size):
        self.networks = [network.clone() for _ in range(size)] # creates a list with shallow copies of the neural network
        return self
    
    def activate_gen(self,X):
        self.outputs = [network.activate(X) for network in self.networks]
        return self.outputs
    
    def get_best(self,*,network_and_loss=False):  # Returns (best_gen_w, best_gen_b) by default and if specified the best network and its loss.
        self.generation_lowest_loss = self.networks[0].loss
        for network in self.networks:
            if network.loss < self.generation_lowest_loss:
                self.generation_lowest_loss = network.loss
                self.best_network = network
        self.best_generation_weights = [layer.best_weights for layer in self.best_network.layers]
        self.best_generation_biases = [layer.best_biases for layer in self.best_network.layers]
        # return the network & loss if the user prompts
        out = (self.best_generation_weights, self.best_generation_biases) + ((self.best_network,self.generation_lowest_loss) if network_and_loss else ())
        return out
    
    def copy_best_network(self,*,network=None):
        if network != None:
            self.best_network = network
        return self.create_gen(self.best_network, self.size)
    
    def mutate_gen(self,*,networks=None):
        for network in self.networks[1:]:   # index [1:] so that the og network is left untouched.
            for layer in network:
                layer.mutate()
        return self.networks

    def update_gen(self,losses=None):
        if type(losses) != np.ndarray:
            if type(losses) is list:
                losses = np.array(losses)
            else:
                raise ValueError(f"'losses' argument expected array, got {type(losses)}")
        if len(losses) != len(self.network.layers):
            raise ValueError(f"'losses' expected array of length {len(self.network.layers)}, got {len(losses)}")
        for network, loss in zip(self.networks,losses):
            network.update(loss)
        
    def save(self,file_name):
        self.networks[0].save(file_name,gen_size=self.size,loss=self.generation_lowest_loss)

def load(file_name) -> Neural_Network | generation: # Load wb from file_name
    with open(file_name,"r") as sf:
        lines = sf.readlines()
        size,loss,activationids,*lines = lines
        size = int(size[:-1])
        loss = float(loss[:-1])
        activationids= list(map(int,list(activationids)[:-1]))
        data = []
        temp = []
        # sort the file data into a list, using "\n" as a splitter
        for line in lines:
            if line == "\n":
                data.append(temp)
                temp = []
                continue
            temp.append(line)
        del temp
    # separate elements of the list into 2D np arrays
    arrays = []
    for li in data:
        list_length = len(li)
        strrow = "".join(li).translate(str.maketrans("","","[\n]"))         # remove characters "[", "]", and "\n" 
        array = np.array([np.fromstring(strrow,sep=" ")])                   # turn the massive string into a 1D list
        array = array.reshape(list_length,int(array.shape[1]/list_length))  # Turn the 1D array into a 2D array using number of lines for dimentions
        arrays.append(array)
    
    sizes = [array.shape for array in arrays[::2]]                      # Get the shapes of the weights, i.e. layer sizes
    sizes = [sizes[0][0]] + [num[1] for num in sizes]                   # Add the input size
    
    NN = Neural_Network()
    NN.create(sizes,activationids)
    NN.set_lowest_loss(loss)
    for i in range(int(len(arrays)/2)):
        NN.layers[i].set_weights(arrays[i*2])
        NN.layers[i].set_biases(arrays[i*2+1])
    if size == 1:
        return NN
    gen = generation(NN,size)
    return gen
    
# basic test for neural network
#================================================================================================================================
if __name__ == "__main__":
    while True:
        try:
            num_inputs = int(input("How many inputs should the neural network recognise?\nEnter a whole number : "))
            if 2 <= num_inputs <= 10:
                break
        finally:
            print("Please enter a number between 2 and 10")
    num_outputs = 3

    X = []
    for i in range(int(2**num_inputs)):                       # iterate through every number and create its binary
        if i%3 != 0:                                          # to challenge the network, skip one third of the numbers
            continue
        str_num = "{:b}".format(i)
        str_num = "0" * (num_inputs - len(str_num)) + str_num # add missing zeros
        X.append(list(map(int, list(str_num))))               # convert to int() and append to X
    X = np.array(X)


    y = []
    for i in X:
        if sum(i) < len(i) / 2:
            y.append(0)
        elif sum(i) > len(i) / 2:
            y.append(2)
        else:
            y.append(1)
    y = np.array(y)

    #print(X,X.shape,y,y.shape,sep="\n\n"); exit() # Use to check on X and y

    options = ["Mostly white", "equally black and white", "mostly black"] 

    #================================================================================================================================
    # network here
    NN = Neural_Network()
    NN.create((num_inputs,4,4,num_outputs),(activations.relu(),activations.relu(), activations.softmax()))


    loss_function = loss_cce()

    best_layer1_weights = NN.layers[0].weights.copy()
    best_layer1_biases  = NN.layers[0].biases.copy()
    best_layer2_weights = NN.layers[1].weights.copy()
    best_layer2_biases  = NN.layers[1].biases.copy()
    best_layer3_weights = NN.layers[2].weights.copy()
    best_layer3_biases  = NN.layers[2].biases.copy() 

    i = 0
    while NN.lowest_loss > 1.01e-07 and i < 100000: #figure out how to implement this loop
        rd_num = randint(-1,1)
        NN.layers[0].weights += (np.zeros((num_inputs,4)) + rd_num)  if i%12==0  else (0.5 * np.random.randn(num_inputs,4))
        NN.layers[0].biases  += (np.zeros((1,4)) + rd_num)           if i%12==1 else (0.5 * np.random.randn(1,4))
        NN.layers[1].weights += (np.zeros((4,4)) + rd_num)           if i%12==2 else (0.5 * np.random.randn(4,4))
        NN.layers[1].biases  += (np.zeros((1,4)) + rd_num)           if i%12==3 else (0.5 * np.random.randn(1,4))
        NN.layers[2].weights += (np.zeros((4,num_outputs)) + rd_num) if i%12==4 else (0.5 * np.random.randn(4,num_outputs))
        NN.layers[2].biases  += (np.zeros((1,num_outputs)) + rd_num) if i%12==5 else (0.5 * np.random.randn(1,num_outputs))

        output = NN.activate(X)
        loss = loss_function.calculate(output,y)
        predictions = np.argmax(output, axis=1)
        accuracy = np.mean(predictions==y)

        if loss < NN.lowest_loss:
            print(f"new set of weights found, iteration:{i} loss:{loss} acc:{accuracy}",end=f"{' '*50}\r")
            best_layer1_weights = NN.layers[0].weights.copy()
            best_layer1_biases  = NN.layers[0].biases.copy()
            best_layer2_weights = NN.layers[1].weights.copy()
            best_layer2_biases  = NN.layers[1].biases.copy()
            best_layer3_weights = NN.layers[2].weights.copy()
            best_layer3_biases  = NN.layers[2].biases.copy()
            NN.lowest_loss = loss
        else:
            NN.layers[0].weights = best_layer1_weights.copy()
            NN.layers[0].biases  = best_layer1_biases.copy() 
            NN.layers[1].weights = best_layer2_weights.copy()
            NN.layers[1].biases  = best_layer2_biases.copy()
            NN.layers[2].weights = best_layer3_weights.copy()
            NN.layers[2].biases  = best_layer3_biases.copy() 
        i+=1
        if i % 20000 == 0:
            print(f"training...(iteration : {i})",end=f"{' '*50}\r")

    brk = False
    while True:
        try:
            usr_input = input(f"Enter a {num_inputs}-digit number: ")
            if len(usr_input) < num_inputs:
                usr_input = usr_input + "0"*(num_inputs-len(usr_input))
            if usr_input[:3] == "end":
                break
            if usr_input[:4] == "save":
                NN.save("tester.txt")
                break
            
            nums = list(map(int,list(usr_input)[:num_inputs]))
        except ValueError:
            print("Error: Please enter an integer.\n")
            continue

        NN.activate(nums)
        print(f"Neural network thinks that the number is '{options[NN.best_node]}' (confidence = {NN.confidence}%).")
    NN.save("tester.txt")