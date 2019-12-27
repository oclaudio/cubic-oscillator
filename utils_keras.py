#utils package
import numpy as np
import tensorflow as tf
from numpy.fft import fft, ifft, fftfreq
from tensorflow.keras import layers
import tensorflow.keras.backend as K

#provides an initial approximation of the noise
def approximate_noise(Y, lam = 10):
   #INPUTS: 
        #Y: data given
        #lam: parameter for control of the linear matrix equation
   
    #get shape of given data
    m,n = Y.shape

    #initialize D as m x m "matrix"
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
    X_smooth = np.vstack([np.linalg.solve(np.eye(m) + lam*D.T.dot(D), Y[:,j].reshape(m,1)).reshape(1,m) for j in range(n)]).astype('float32').T
    
    #estimated noise
    N_hat = Y-X_smooth
    
    #OUTPUTS:
        #N_hat: initial noise approximation
        #X_smooth: initial state approximation
    return N_hat, X_smooth


#Explicit Runge-Kutta time integrator.  Assumes no time dependence in f
def RK_timestepper(x,f,h,direction='F'):
    #INPUTS:
        #x: vector or matrix of network input states; usually given as a placeholder
        #f: the constructed neural network 
        #h: the (constant) time step 
        #direction: either 'F' or  'B': forward/backward direction of the Runge-Kutta scheme, defaults to forward
    
    #RK4-method
    b = [1/6,1/3,1/3,1/6]
    A = [[],[1/2],[0, 1/2],[0,0,1]]
    
    #number of steps for the Runge-Kutta scheme; corresponds to p
    steps = len(b)
    
    #forward direction
    if direction == 'F':
        #initiate k as list, first element being the network evaluated on x
        k = [f(x)]
        for i in range(1,steps):
            #add other summands to the list
            k.append(f(tf.add_n([x]+[h*A[i][j]*k[j] for j in range(i) if A[i][j] != 0])))
    #backward direction, changes only the sign before f
    else:        
        k = [-f(x)]
        for i in range(1,steps):
            k.append(-f(tf.add_n([x]+[h*A[i][j]*k[j] for j in range(i) if A[i][j] != 0])))
            
    #OUTPUT:
        #Runge-Kutta scheme for the given inputs
    return tf.add_n([x]+[h*b[j]*k[j] for j in range(steps)])


##Explicit Runge-Kutta time integrator for Keras without use of TensorFlow for improved computation time. 
def RK_timestepper_keras(t,f,h,direction='F',method = 'RK4'):
    #INPUTS:
        #x: vector or matrix of network input states; usually given as a placeholder
        #f: the constructed neural network 
        #h: the (constant) time step 
        #direction: either 'F' or  'B': forward/backward direction of the Runge-Kutta scheme, defaults to forward
        
    #RK4-method
    b = [1/6,1/3,1/3,1/6]
    A = [[],[1/2],[0, 1/2],[0,0,1]]
    
    #initialize k as a list
    k = [f(t, steps=1)]
    
    #add other summands to the list
    for i in range(1,len(b)):
        k.append(f(np.add([t],[h*A[i][j]*k[j] for j in range(i) if A[i][j] != 0]).reshape((1,2)), steps=1))
        
    #save sum in a list
    summ = [[0,0]]
    
    #calculate sum
    for i in range(len(b)):
        summ = np.add([summ],[h*b[i]*k[i]]).flatten()
        
    #OUTPUT:
        #Runge-Kutta scheme for the given inputs   
    return np.add([[t]], summ)


#class for calculation of the loss 
class loss_class(object):
    def __init__(self, Y, num_dt, dt, model, gamma, beta, decay_const, N_init,N=None, scope='N', **N_kwargs):
        #INPUTS:
            #Y: given measurements
            #num_dt: denotes q
            #dt: time-step between each measurement
            #model: Keras-model of the network
            #gamma: hyperparameter of noise regularizer term, defaults to 1e-5
            #beta: hyperparameter of weights regularizer term, defaults to 1e-8
            #decay_const: hyperparameter \omega_0 for exponential decay for weights of the loss function, defaults to 0.9
            #N_init: initial noise approximation
            #N: noise parameter
            #scope: noise
            #N_kwargs: additional arguments; not used
        
        #define self
        self.scope = scope
        self.num_dt = num_dt
        self.dt = dt
        self.model = model
        self.Y = Y
        self.decay_const = decay_const
        self.gamma = gamma
        self.beta = beta
        
        #initialize N as K variable
        with tf.name_scope(self.scope):
            if N is None:
                N = K.variable(N_init, dtype=tf.float32,
                                    name='N', **N_kwargs)
            self.N_variable = N
            self.N = N
    
    #loss function
    def loss(self, y_true, y_pred):
        #INPUTS: 
            #y_true, y_pred: necessary for use of loss function in Keras, not actually used here            
            
        #get necessary variables
        Y = self.Y
        m,n = Y.shape
        model = self.model
        dt = self.dt
        num_dt = self.num_dt
        gamma = self.gamma
        beta = self.beta
        decay_const = self.decay_const
        
        #compute loss
        with tf.name_scope(self.scope):
            
            #noisy measurements for y_{j+i}, i=0 and j=q+1,...,m-q
            Y_0 = Y[num_dt:m-num_dt,:]
            #noise
            N = self.N_variable
            #lists for forward and backward Y
            true_forward_Y = []
            true_backward_Y = []
            
            #fill forward and backward lists with correct measurements
            for j in range(num_dt):
                true_forward_Y.append(Y[num_dt+j+1:m-num_dt+j+1,:])
                true_backward_Y.append(Y[num_dt-j-1:m-num_dt-j-1,:])
                
                
            #compute y_j - noise for j in [q+1, m-q]
            X_0 = tf.subtract(Y_0, tf.slice(N, [num_dt,0],[m-2*num_dt,n]))  
            
            #apply the Runge-Kutta scheme, once for both directions, for the given network and X_0
            pred_forward_X = [RK_timestepper(X_0, model.__call__, dt)]
            pred_backward_X = [RK_timestepper(X_0,  model.__call__, dt,direction = 'B')]
    
            #apply the Runge-Kutta scheme for both directions up to a total number of q times, save all in a list
            for j in range(1,num_dt):
                pred_forward_X.append(RK_timestepper(pred_forward_X[-1],  model.__call__, dt))
                pred_backward_X.append(RK_timestepper(pred_backward_X[-1],  model.__call__, dt, direction = 'B'))
                
                
            #  Forward and backward predictions of measured (noisy) state

            #add estimated noise at time j+i to the results of the Runge-Kutta scheme above
            pred_forward_Y = [pred_forward_X[j] + tf.slice(N, [num_dt+1+j,0],[m-2*num_dt,n]) for j in range(num_dt)]
            pred_backward_Y = [pred_backward_X[j] +tf.slice(N, [num_dt-1-j,0],[m-2*num_dt,n]) for j in range(num_dt)]
            
    
    
            #  Set up cost function
        
            #exponentially decreasing importance 
            output_weights = [decay_const**j for j in range(num_dt)] 

        
            forward_fidelity = tf.reduce_sum([w*tf.losses.mean_squared_error(true,pred) \
                                for (w,true,pred) in zip(output_weights,true_forward_Y,pred_forward_Y)])

            backward_fidelity = tf.reduce_sum([w*tf.losses.mean_squared_error(true,pred) \
                                for (w,true,pred) in zip(output_weights,true_backward_Y,pred_backward_Y)])

            fidelity = tf.add(forward_fidelity, backward_fidelity)
    
            #get weights of the network
            weights = []
            for layers in model.layers[1:]:
                weights.append(layers.get_weights()[0])
                
            #calculate weights and noise regularizer term
            weights_regularizer =  tf.reduce_mean([tf.nn.l2_loss(W) for W in weights])     
            noise_regularizer = tf.nn.l2_loss(N)
            #calculate loss function
            cost = tf.reduce_sum(fidelity  + gamma*noise_regularizer+ beta*weights_regularizer)
            
            return cost
            
          

