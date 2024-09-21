import numpy as np
from random import randint
class Neural_Network:
    def __init__(self, *, interrupt=None):
        self.loss_function = None
        #if not callable(interrupt):
        #    raise NameError(f"Neural Network: expected 'inturrupt' function, got {type(interrupt)}")
        self.interrupt = interrupt
        self.lowest_loss = 999999999999
        self.iteration = 0
    class neuron_layer:
        def __init__(self, n_inputs, n_neurons, activation):
            self.best_weights = None
            self.best_biases  = None
            self.n_inputs = n_inputs
            self.n_neurons = n_neurons
            self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))
            self.activation = activation
        def set(self,weights,biases): # Manually set w/b
            self.best_weights = weights
            self.best_biases = biases
        def mutate(self):
            self.weights += 0.10*np.random.randn(self.n_inputs, self.n_neurons)
            self.biases += 0.10*np.random.randn(1, self.n_neurons)
        def forward(self, inputs):
            self.activation_input = np.dot(inputs, self.weights) + self.biases
            self.activate(self.activation_input)
        def activate(self,input):
            self.activation.forward(input)
            self.output = self.activation.output


    class activation_step:
        def forward(self, inputs):
            vstep = np.vectorize(lambda x: 1 if x > 0 else 0)
            self.output = vstep(inputs)
    
    class activation_relu:
        def forward(self, inputs):
            self.output = np.maximum(0,inputs)

    class activation_softmax:
        def __init__(self):
            self.loss_function = Neural_Network.loss_cce().forward
        def forward(self, inputs):
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            self.output = probs
        

    class loss:
        def calculate(self, output, y):
            sample_losses = self.forward(output,y) 
            data_loss = np.mean(sample_losses)
            return data_loss

    class loss_cce(loss):
        def forward(self, y_pred, y_true):
            samples = len(y_pred)
            y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
            if len(y_true.shape) == 1:
                correct_confidences = y_pred_clipped[range(samples), y_true]
            elif len(y_true.shape) == 2:
                correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            negative_log_probs = -np.log(correct_confidences)
            return negative_log_probs
    

    def create(self,*args):
        if len(args) != 2:
            raise ValueError(f"create((layers), (activations)) expected 2 positional arguments, got {len(args)}")
        self.sizes, self.activations = args
        
        # Validation
        # Ensuring logical amounts have been entered
        for num in self.sizes:
            if type(num) != int:
                raise ValueError(f"function create() expected int, got {type(num)}.")

            if num < 1:
                raise ValueError("Layer size cannot be less than 1.")

        #Coding begins!
        self.layers = []
        for i in range(len(self.sizes)-2):        # minus 1 bc input layer doesnt require an object
            layer = self.neuron_layer(self.sizes[i], self.sizes[i+1],self.activations[i])
            self.layers.append(layer)
        self.layers.append(self.neuron_layer(self.sizes[-2], self.sizes[-1],self.activations[-1])) # create a list w all the layers


        for layer in self.layers: # prepare to evolve
            layer.best_weights = layer.weights.copy()
            layer.best_biases  = layer.biases.copy()
    
    def activate(self,X):
        if X.shape[1] != self.sizes[0]:
            raise ValueError(f"Target inputs have a different shape from NN inputs: {X.shape} vs {num_inputs}") # validating X
        mailman = X # it's called the mailman because it passes the values from one object to the next
        for layer in self.layers:
            layer.forward(mailman)
            mailman = layer.output
        return mailman
    
    def update(self,loss,*,verbose=False):
        self.iteration += 1
        if loss < self.lowest_loss:
            for layer in self.layers: # update all w/b
                layer.best_weights = layer.weights.copy()
                layer.best_biases  = layer.biases.copy()
            self.lowest_loss = loss
            if verbose:
                print(f"new set of weights found, iteration:{self.iteration} loss:{loss}")
        else:
            for layer in self.layers: # revert all w/b
                layer.weights = layer.best_weights.copy()
                layer.biases  = layer.best_biases.copy()
    def save(self,file_name):
        with open(file_name,"w") as sf:
            data = []
            for layer in NN.layers:
                data.extend((str(layer.best_weights),"\n\n",str(layer.best_biases),"\n\n"))
            del data[-2:]
            sf.writelines(data)
    def load(file_name)

class generation:
    def __init__(self,size,network):
        self.size = size
        self.network = network
        # define generation best w/b by creating a list w all the layers and their weights/biases.
    # calculate loss for all the layers
    # keep the best w/b and assign to all the networks
    # mutate
    # r&r


#================================================================================================================================
# defining variables/parameters
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
    if i%3 != 0:                                          # to make things interesting, skip one third of the numbers
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

print(X,X.shape,y,y.shape,sep="\n\n"); #exit() # Use to check on X and y

options = ["Mostly white", "equally black and white", "mostly black"] 

#================================================================================================================================
# network here
NN = Neural_Network()
NN.create((num_inputs,4,4,num_outputs),(NN.activation_relu(),NN.activation_relu(), NN.activation_softmax()))


loss_function = NN.loss_cce()

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
        print(f"new set of weights found, iteration:{i} loss:{loss} acc:{accuracy}\r")
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
        print(f"training...(iteration : {i})",end="\r")

brk = False
while True:
    try:
        usr_input = input(f"Enter a {num_inputs}-digit number: ")
        if len(usr_input) < num_inputs:
            usr_input = usr_input + "0"*(num_inputs-len(usr_input))
        if usr_input[:3] == "end":
            break
        nums = np.array([list(map(int,list(usr_input)[:num_inputs]))])
    except ValueError:
        print("Error: Please enter an integer.\n")
        continue

    AI_answer = NN.activate(nums)
    best_node = np.argmax(AI_answer)
    confidence = np.floor( max(AI_answer[0]) * 10000 ) / 100 
    print(f"Neural network thinks that the number is '{options[best_node]}' (confidence = {confidence}%).")

NN.save("Test_file")

    

