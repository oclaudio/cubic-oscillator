#necessary libraries
import numpy as np
import tensorflow as tf
from numpy.fft import fft, ifft, fftfreq


#Explicit Runge-Kutta time integrator.  Assumes no time dependence in f
def RK_timestepper(x,t,f,h,weights,biases,direction='F',method = 'RK4'):
    #INPUTS:
        #x: vector or matrix of network input states; usually given as a placeholder
        #t: vector of time; since no dependency on time, could be dropped
        #f: the constructed neural network 
        #h: the (constant) time step 
        #weights: the weights of the network
        #biases: the biases of the network
        #direction: either 'F' or  'B': forward/backward direction of the Runge-Kutta scheme, defaults to forward
        #method: used method; defaults to RK4
    
    #several possible Runge-Kutta methods
    if method == 'RK4_38':
        b = [1/8,3/8,3/8,1/8]
        A = [[],[1/3],[-1/3, 1],[1,-1,1]]

    elif method == 'Euler':
        b = [1]
        A = [[]]

    elif method == 'Midpoint':
        b = [0,1]
        A = [[],[1/2]]

    elif method == 'Heun':
        b = [1/2,1/2]
        A = [[],[1]]

    elif method == 'Ralston':
        b = [1/4,3/4]
        A = [[],[2/3]]

    elif method == 'RK3':
        b = [1/6,2/3,1/6]
        A = [[],[1/2],[-1,2]]
        
    #defaults to RK4 method
    else:
        b = [1/6,1/3,1/3,1/6]
        A = [[],[1/2],[0, 1/2],[0,0,1]]
    
    #number of steps for the Runge-Kutta scheme; corresponds to p
    steps = len(b)

    #forward direction
    if direction == 'F':
        #initiate K as list, first element being the network evaluated on x, weights, biases
        K = [f(x, weights, biases)]
        for i in range(1,steps):
            #add other summands to the list
            K.append(f(tf.add_n([x]+[h*A[i][j]*K[j] for j in range(i) if A[i][j] != 0]), weights, biases))
            
    #backward direction, changes only the sign before f
    else:
        K = [-f(x, weights, biases)]
        for i in range(1,steps):
            K.append(-f(tf.add_n([x]+[h*A[i][j]*K[j] for j in range(i) if A[i][j] != 0]), weights, biases))
    
    #OUTPUT:
        #Runge-Kutta scheme for the given inputs
    return tf.add_n([x]+[h*b[j]*K[j] for j in range(steps)])

#applies RK4-method using RK_timestepper in forward direction
def RK4_forward(x,t,f,h,weights,biases):
    #INPUTS:
        #as in RK_timestepper
    
    #OUTPUTS:
        #forward timestepper scheme using RK4 method
    return RK_timestepper(x,t,f,h,weights,biases,direction='F',method = 'RK4_classic')

#applies RK4-method using RK_timestepper in backward direction 
def RK4_backward(x,t,f,h,weights,biases):
    #INPUTS: 
        #same as for RK_timestepper 
    
    #OUTPUTS:
        #backward timestepper scheme using RK4 method
    return RK_timestepper(x,t,f,h,weights,biases,direction='B',method = 'RK4_classic')

#calculates the activation of the next layer given weights and biases of previous layer
def dense_layer(x, W, b, last = False):
    #INPUTS:
        #x: Unit of the previous layer
        #W: Weights of the current layer
        #b: Biases of the current layer
        #last: True if on last layer, false otherwise
    
    #x -> W*x
    x = tf.matmul(W,x)
    #x -> x+b
    x = tf.add(x,b)
    
    #OUTPUTS:
        #if last: output the activation of output layer using identity activation function
        #otherwise: output the activation of the current layer using ELu activation function
    if last: return x
    else: return tf.nn.elu(x)

#construction of the network
def simple_net(x, weights, biases):
    #INPUTS:
        #x: vector or matrix of network input states; usually given as a placeholder
        #weights: weights of the network
        #biases: biases of the network
    
    #initialize input layer as list 
    layers = [x]
    
    #loop excludes last layer of the NN
    for l in range(len(weights)-1):
        #add hidden layers to the list, using dense_layer above
        layers.append(dense_layer(layers[l], weights[l], biases[l]))

    #output layer, linear activation function
    out = dense_layer(layers[-1], weights[-1], biases[-1], last = True)
    
    #OUTPUTS:
        #returns activation of output layer
    return out

#initial noise approxmation
def approximate_noise(Y, lam = 10):
    #INPUTS: 
        #Y: data given
        #lam: parameter for control of the linear matrix equation
   
    #get shape of given data
    n,m = Y.shape

    #initialize D as m x m matrix
    D = np.zeros((m,m))
    #Set elements of D
    D[0,:4] = [2,-5,4,-1]
    D[m-1,m-4:] = [-1,4,-5,2]
    for i in range(1,m-1):
        D[i,i] = -2
        D[i,i+1] = 1
        D[i,i-1] = 1

    #D^2
    D = D.dot(D)

    #solve (I + lam*D^T*D)*x = Y[j,:], reshape to correct dimensions 
    X_smooth = np.vstack([np.linalg.solve(np.eye(m) + lam*D.T.dot(D), Y[j,:].reshape(m,1)).reshape(1,m) for j in range(n)])

    #estimated noise
    N_hat = Y-X_smooth

    #OUTPUTS:
        #N_hat: initial noise approximation
        #X_smooth: initial state approximation
    return N_hat, X_smooth

#initializes the network weights, biases and noise as TensorFlow variables
def get_network_variables(n, n_hidden, size_hidden, N_hat):
    #INPUT:
        #n: dimension of the measurements
        #n_hidden: number of hidden layers
        #size_hidden: size of every hidden layer
        #N_hat: initial noise approximation
 
    #compute layer sizes, n nodes in input and output layer, size_hidden nodes in n_hidden layers
    layer_sizes = [n] + [size_hidden for _ in range(n_hidden)] + [n]
    
    #compute number of layers
    num_layers = len(layer_sizes)

    #initialization of network weights and biases
    weights = []
    biases = []
    for j in range(1,num_layers):
        #add weights of layer j to weights list using Xavier initializer
        weights.append(tf.get_variable("W"+str(j), [layer_sizes[j],layer_sizes[j-1]], \
                                       initializer = tf.contrib.layers.xavier_initializer(seed = 1)))
        #add biases of layer j to biases list using zeros initializer
        biases.append(tf.get_variable("b"+str(j), [layer_sizes[j],1], initializer = tf.zeros_initializer()))

    #create TensorFlow variable N with initializer N_hat as float32
    #in particular, this turns N into a trainable parameter
    N = tf.get_variable("N", initializer = tf.cast(N_hat, dtype = tf.float32))

    #OUTPUTS:
        #returns the initialized weights, biases and noise as TensorFlow variables
        #all returned variables are trainable
    return (weights, biases, N)

#defines the loss function
def create_computational_graph(n, N_hat, net_params,num_dt = 10, method = 'RK4', gamma = 1e-5, beta = 1e-8, weight_decay = 'exp', decay_const = 0.9):
    #INPUTS:
        #n: dimension of the measurements
        #N_hat: initial noise approximation
        #net_params: network parameters, given as (weights, biases, N)
        #num_dt: denotes q
        #method: the method for the Runge-Kutta scheme in RK_timestepper
        #gamma: hyperparameter of noise regularizer term, defaults to 1e-5
        #beta: hyperparameter of weights regularizer term, defaults to 1e-8
        #weights_decay: either 'linear' or 'exp', weights for the loss function, defaults to 'exp'
        #decay_const: hyperparameter \omega_0 for exponential decay for weights of the loss function, defaults to 0.9
    
    #n should be equal to first dimension of N_hat
    assert(n == N_hat.shape[0])
    
    #set m to second dimension of N_hat
    m = N_hat.shape[1]

   
    # Placeholders for initial condition
    
    #set up placeholders for y_{j+i}, i=0 and j=q+1,...,m-q; noise measurements
    Y_0 = tf.placeholder(tf.float32, [n,None], name = "Y_0") 
    #corresponding time
    T_0 = tf.placeholder(tf.float32, [1,None], name = "T_0")

    
    # Placeholders for true forward and backward predictions
    
    true_forward_Y = []
    true_backward_Y = []

    for j in range(num_dt):
        #add placeholders for rest of the noisy measurements y_{j+i}, i in [-q,q]\{0}
        true_forward_Y.append(tf.placeholder(tf.float32, [n,None], name = "Y"+str(j+1)+"_true"))
        true_backward_Y.append(tf.placeholder(tf.float32, [n,None], name = "Yn"+str(j+1)+"_true"))

    #placeholder for timestep h
    h = tf.placeholder(tf.float32, [1,1], name = "h")

    
    #  Forward and backward predictions of true state

    #get the network parameters
    (weights, biases, N) = net_params
    
    #compute y_j - noise for j in [q+1, m-q]
    X_0 = tf.subtract(Y_0, tf.slice(N, [0,num_dt],[n,m-2*num_dt]))

    #apply the Runge-Kutta scheme, once for both directions, for the given network and X_0
    pred_forward_X = [RK_timestepper(X_0, T_0, simple_net, h, weights, biases, method = method)]
    pred_backward_X = [RK_timestepper(X_0, T_0, simple_net, h, weights, biases, method = method, direction = 'B')]
    
    #apply the Runge-Kutta scheme for both directions up to a total number of q times, save all in a list
    for j in range(1,num_dt):
        pred_forward_X.append(RK_timestepper(pred_forward_X[-1], T_0, simple_net, h, weights, biases, method = method))
        pred_backward_X.append(RK_timestepper(pred_backward_X[-1], T_0, simple_net, h, weights, biases,\
                                            method = method, direction = 'B'))
      
    
    #  Forward and backward predictions of measured (noisy) state

    #add estimated noise at time j+i to the results of the Runge-Kutta scheme above
    pred_forward_Y = [pred_forward_X[j] + tf.slice(N, [0,num_dt+1+j],[n,m-2*num_dt]) for j in range(num_dt)]
    pred_backward_Y = [pred_backward_X[j] + tf.slice(N, [0,num_dt-1-j],[n,m-2*num_dt]) for j in range(num_dt)]


    
    #  Set up cost function
    
    #defaults to exponential decay
    if weight_decay == 'linear': output_weights = [(1+j)**-1 for j in range(num_dt)]
    else: output_weights = [decay_const**j for j in range(num_dt)]
   
    forward_fidelity = tf.reduce_sum([w*tf.losses.mean_squared_error(true,pred) \
                          for (w,true,pred) in zip(output_weights,true_forward_Y,pred_forward_Y)])

    backward_fidelity = tf.reduce_sum([w*tf.losses.mean_squared_error(true,pred) \
                          for (w,true,pred) in zip(output_weights,true_backward_Y,pred_backward_Y)])

    fidelity = tf.add(forward_fidelity, backward_fidelity)

    
    # Regularizer for NN weights
    weights_regularizer = tf.reduce_mean([tf.nn.l2_loss(W) for W in weights])

    # Regularizer for explicit noise term
    noise_regularizer = tf.nn.l2_loss(N)

    # Weighted sum of individual cost functions
    cost = tf.reduce_sum(fidelity + beta*weights_regularizer + gamma*noise_regularizer)

    # BFGS optimizer via scipy
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(cost, options={'maxiter': 50000, 
                                                                      'maxfun': 50000,
                                                                      'ftol': 1e-15, 
                                                                      'gtol' : 1e-11,
                                                                      'eps' : 1e-12,
                                                                      'maxls' : 100})

    #placeholders used above
    placeholders = {'Y_0': Y_0,
                    'T_0': T_0,
                    'true_forward_Y': true_forward_Y,
                    'true_backward_Y': true_backward_Y,
                    'h': h}

    return optimizer, placeholders

