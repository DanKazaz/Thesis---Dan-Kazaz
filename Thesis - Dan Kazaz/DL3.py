import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

class DLModel:
    def __init__(self, name="Model"):
        self.layers = [None]  # אתחול ריק כדי שהשכבה הראשונה תהיה באינדקס 1
        self._is_compiled = False
        self.name = name
        
    def add(self, layer):
        self.layers.append(layer)
        
    def squared_means(self, AL, Y):
        squared_diff = np.square(AL - Y)
        return squared_diff
    
    def squared_means_backward(self, AL, Y):
        squared_diff = 2 * (AL - Y)
        return squared_diff
    
    def cross_entropy(self, AL, Y):
        AL = np.where(AL==0, 1e-10, AL)
        AL = np.where(AL==1, 1-1e-10, AL)
        cross_entropy_diff = np.where(Y==0, -np.log(1-AL), -np.log(AL))
        return cross_entropy_diff
 
    def cross_entropy_backward(self, AL, Y):
        AL = np.where(AL==0, 1e-10, AL)
        AL = np.where(AL==1, 1-1e-10, AL)
 
        dJ_dAL = np.where(Y == 0, 1/(1-AL),-1/AL)
        return dJ_dAL
    
    def _categorical_cross_entropy(self, AL, Y):
        eps=1e-10
        L=np.where(AL==0,eps,AL)
        L=np.where(L==1,1-eps,L)
        errors = np.where(Y == 1, -np.log(AL), 0)
        return errors
    

    def _categorical_cross_entropy_backward(self, AL, Y):
        dZ = AL - Y
        return dZ
    
    def compile(self, loss, threshold=0.5):
        if loss == "squared_means":
            self.loss = loss
            self.loss_forward = self.squared_means
            self.loss_backward = self.squared_means_backward
        elif loss == "cross_entropy":
            self.loss = loss
            self.loss_forward = self.cross_entropy
            self.loss_backward = self.cross_entropy_backward
        elif loss == "categorical_cross_entropy":
            self.loss = loss
            self.loss_forward = self._categorical_cross_entropy
            self.loss_backward = self._categorical_cross_entropy_backward
            
        self.threshold = threshold
        self._is_compiled = True
        
    def compute_cost(self, AL, Y):
        m = Y.shape[0]
        error = self.loss_forward(AL, Y)
        cost = np.sum(error) / m
        return cost
    
    def train(self, X, Y, num_iterations):
        print_ind = max(num_iterations // 100, 1)
        L = len(self.layers)
        costs = []
        for i in range(num_iterations):
            # forward propagation
            Al = X
            for l in range(1,L):
                Al = self.layers[l].forward_propagation(Al,False)
            #backward propagation
            dAl = self.loss_backward(Al, Y)
            for l in reversed(range(1,L)):
                 dAl = self.layers[l].backward_propagation(dAl)
                # update parameters
                 self.layers[l].update_parameters()
            #record progress
            if i % print_ind == 0:
                J = self.compute_cost(Al, Y)
                costs.append(J)
                print("cost after ", i // print_ind, "%:", J)
        return costs

    def predict(self, X):
        L = len(self.layers)
        Al = X
        for l in range(1, L):
            Al = self.layers[l].forward_propagation(Al, is_predict=True)
            
        result = (Al > self.threshold).astype(bool)
        return result
    
    def __str__(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers)-1) +"\n"
        if self._is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) +"\n"
            s += "\t\tloss function: " + self.loss + "\n\n"

        for i in range(1,len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"
        return s
    
    def save_weights(self, path):
        for i in range(1, len(self.layers)):
            self.layers[i].save_weights(path, self.layers[i].name)

class DLLayer:
    def __init__(self, name, num_units, input_shape, activation="relu", W_initialization="random", learning_rate=1.2, optimization=None, leaky_relu_d=None):
        self.name = name
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation
        self.alpha = learning_rate
        self._optimization = optimization
        self.leaky_relu_d = leaky_relu_d
        self.W_initialization = W_initialization
        self.random_scale = 0.01

        # Initialize adaptive alpha parameters
        if self._optimization == "adaptive":
            self._adaptive_alpha_b = np.full((self._num_units, 1), self.alpha)
            self._adaptive_alpha_W = np.full((self._num_units, *self._input_shape), self.alpha)
            self._adaptive_switch = 0.5
            self._adaptive_cont = 1.1  

        # Initialize parameters using the init_weights method
        self.init_weights(W_initialization)

        # Initialize activation_forward based on the activation type
        if activation == "sigmoid":
            self.activation_forward = self._sigmoid
        elif activation == "tanh":
            self.activation_forward = self._tanh
        elif activation == "trim_sigmoid":
            self.activation_forward = self._trim_sigmoid
        elif activation == "trim_tanh":
            self.activation_forward = self._trim_tanh
        elif activation == "relu":
            self.activation_forward = self._relu
        elif activation == "leaky_relu":
            self.activation_forward = self._leaky_relu
        elif activation == "softmax":
            self.activation_forward = self._softmax

        # Initialize activation_trim
        self.activation_trim = 1e-10
            
    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_units, 1), dtype=float)

        if W_initialization == "zeros":
            self.W = np.zeros((self._num_units, *self._input_shape))
        elif W_initialization == "random":
            self.random_scale = 0.01 # or 0.1, 0.01 etc.
            self.W = np.random.randn(self._num_units, *self._input_shape) * self.random_scale
        elif W_initialization == "He":
            self.W = np.random.randn(self._num_units, *self._input_shape) * np.sqrt(1 / self._input_shape[0])
        elif W_initialization == "Xaviar":
            self.W = np.random.randn(self._num_units, *self._input_shape) * np.sqrt(2 / self._input_shape[0])
        else:  
            try:
                with h5py.File(W_initialization, 'r') as hf:
                    self.W = hf['W'][:]
                    self.b = hf['b'][:]
            except (FileNotFoundError):
                raise NotImplementedError("Unrecognized initialization:", W_initialization)
            
    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
         
    def _tanh(self, Z):
        return np.tanh(Z)

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _leaky_relu(self, Z):
        return np.where(Z > 0, Z, self.leaky_relu_d * Z)

    def _trim_sigmoid(self, Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1 / (1 + np.exp(-Z))
            except FloatingPointError:
                Z = np.where(Z < -100, -100, Z)
                A = 1 / (1 + np.exp(-Z))

            TRIM = self.activation_trim

            if TRIM > 0:
                A = np.where(A < TRIM, TRIM, A)
                A = np.where(A > 1 - TRIM, 1 - TRIM, A)

        return A

    def _trim_tanh(self, Z):
        A = np.tanh(Z)

        TRIM = self.activation_trim

        if TRIM > 0:
            A = np.where(A < -1 + TRIM, TRIM, A)
            A = np.where(A > 1 - TRIM, 1 - TRIM, A)

        return A
    
    def _softmax (self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)
    
    def forward_propagation(self, A_prev, is_predict):
        # Save A_prev for backpropagation
        self._A_prev = np.array(A_prev, copy=True)

        # Calculate linear part: Z = W * A_prev + b
        self._Z = np.dot(self.W, A_prev) + self.b

        # Apply activation function to get A
        A = self.activation_forward(self._Z)

        return A

    def _sigmoid_backward(self, dA):
        A = self._sigmoid(self._Z)
        dZ = dA * A * (1 - A)
        return dZ

    def _tanh_backward(self, dA):
        A = self._tanh(self._Z)
        dZ = dA * (1 - A**2)
        return dZ

    def _relu_backward(self, dA):
        dZ = np.where(self._Z <= 0, 0, dA)
        return dZ

    def _leaky_relu_backward(self, dA):
        dZ = np.where(self._Z <= 0, self.leaky_relu_d * dA, dA)
        return dZ

    def _softmax_backward(self, dZ):
        return dZ

    def activation_backward(self, dA):
        if self._activation == "sigmoid":
            return self._sigmoid_backward(dA)
        elif self._activation == "trim_sigmoid":
            return self._sigmoid_backward(dA)
        elif self._activation == "tanh":
            return self._tanh_backward(dA)
        elif self._activation == "relu":
            return self._relu_backward(dA)
        elif self._activation == "leaky_relu":
            return self._leaky_relu_backward(dA)
        elif self._activation == "softmax":
            return self._softmax_backward(dA)
        
    def backward_propagation(self, dA):
        """
        Implement the backward propagation for the current layer.

        Arguments:
        dA -- post-activation gradient for the current layer A
        """
        m = dA.shape[1]  # number of examples in the batch
        A_prev = self._A_prev

        # Compute dZ using the activation_backward method
        dZ = self.activation_backward(dA)

        # Compute gradients
        self.dW = (1/m) * (dZ @ A_prev.T)
        self.db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = self.W.T @ dZ

        return dA_prev
    
    def update_parameters(self):
        if self._optimization == "adaptive":
            self._adaptive_alpha_W = np.where(self._adaptive_alpha_W * self.dW > 0, self._adaptive_alpha_W * self._adaptive_cont, self._adaptive_alpha_W * self._adaptive_switch)
            self._adaptive_alpha_b = np.where(self._adaptive_alpha_b * self.db > 0, self._adaptive_alpha_b * self._adaptive_cont, self._adaptive_alpha_b * self._adaptive_switch)
            self.W -= self._adaptive_alpha_W * self.dW
            self.b -= self._adaptive_alpha_b * self.db
        else:
            self.W -= self.alpha * self.dW
            self.b -= self.alpha * self.db

    def __str__(self):
        s = self.name + " Layer:\n"
        s += "\tnum_units: " + str(self._num_units) + "\n"
        s += "\tactivation: " + self._activation + "\n"
        if self._activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d)+"\n"
        s += "\tinput_shape: " + str(self._input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self.alpha) + "\n"
        #optimization
        if self._optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            s += "\t\t\tcont: " + str(self._adaptive_cont)+"\n"
            s += "\t\t\tswitch: " + str(self._adaptive_switch)+"\n"
        # parameters
        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.W.shape)+"\n"
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()
        return s;

    def save_weights(self, path, file_name):
        if not os.path.exists(path):
            os.makedirs(path)
    
        with h5py.File(path+"/"+file_name+'.h5', 'w') as hf:
            hf.create_dataset("W", data=self.W)
            hf.create_dataset("b", data=self.b)
       


