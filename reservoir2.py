import numpy as np
import scipy.sparse


class Reservoir:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, 
	learning_rate, spectral_radius, connectivity):
        """ Initializes the values of the network """
        
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        self.spectral = spectral_radius
        self.density = connectivity



        # Initializes weights with sparse density determined by connectivity.
        self.wih = scipy.sparse.random(self.hnodes, self.inodes, 
            density = self.density).todense()
        #self.h_past = np.zeros(self.hnodes)
        self.wh = scipy.sparse.random(self.hnodes, self.hnodes, 
            density = .10).todense()
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), \
			(self.onodes, self.hnodes))
        self.h = np.zeros((self.wh.shape[0], 1))
        self.final_outputs = np.zeros((self.who.shape[0], 1))
        
        
        self.eigen_val, self.eigen_vec = np.linalg.eig(self.wh)
        self.wh_max = np.max(np.abs(self.eigen_val))
        self.wh /= (np.abs(self.wh_max) / self.spectral)
        
        
        # Anon function to call sigmoid function.
        #self.sigmoid = lambda x : (1 / (1 + np.exp(-x)))
        self.sigmoid = lambda x : np.tanh(x)


    def train(self, inputs_list, targets_list):
        """ Takes the inputs and target values for a network and forward
        propagates the data. It then compares target value with actual value
        and propagates the errors back through the nodes, updating weights.
        """

        
        #last_h = {0 : h}

        # Creates 2-dimensional arrays from lists. Will prepend 1s to the 
		# array if not at least ndim=2.
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        '''
		# Calcs dot product of input-hidden weights and input values, performs
        # sigmoid acting as first layer of reservoir.
        #reservoir_inputs = np.dot(self.wih, inputs) + np.dot(self.h_past, self.h)
        #current_reservoir = self.sigmoid(reservoir_inputs)
        reservoir_inputs = np.dot(self.wih, inputs)
        reservoir_inputs = np.dot(self.wh, reservoir_inputs)
        # Processes through past iteration of reservoir.
        #reservoir_inputs = np.dot(self.wh2, reservoir_inputs)
        reservoir_calcs = self.sigmoid(reservoir_inputs)

		# Calcs dot product of hidden-output weights and input values, outputs
		# an array w/ summed output values for each onode. Applies sigmoid.
        final_inputs = np.dot(self.who, reservoir_calcs)
        
        final_outputs = self.sigmoid(final_inputs)
		'''
        '''
        self.h = self.wih @ inputs + self.wh @ self.h
        self.h = self.sigmoid(self.h)
        self.final_outputs = self.who @ self.h
        #self.final_outputs = self.sigmoid(self.final_outputs)
        '''
        self.h = self.wih @ inputs
        self.h += self.wh @ self.h
        self.h = self.sigmoid(self.h)
        self.final_outputs = self.who @ self.h
        self.final_outputs = self.sigmoid(self.final_outputs)

        # Calcs error between real answer and NN prediction.
        output_errors = targets - self.final_outputs

		# Updates hidden-output and input-hidden weights based on error.
        self.who += self.lr * np.dot((output_errors * self.final_outputs * \
			(1.0 - self.final_outputs)), np.transpose(self.h))

        return output_errors


    def query(self, inputs_list):
        """ Forward propagates validation data through the network to check
        performance. Check train function above for comments on how commands
        below function.
        """

        inputs = np.array(inputs_list, ndmin=2).T

        self.h = self.wih @ inputs
        self.h += self.wh @ self.h
        self.h = self.sigmoid(self.h)
        self.final_outputs = self.who @ self.h
        self.final_outputs = self.sigmoid(self.final_outputs)
        '''
        self.h = self.wih @ inputs + self.wh @ self.h
        self.h = self.sigmoid(self.h)
        self.final_outputs = self.who @ self.h
        self.final_outputs = self.sigmoid(self.final_outputs)
        '''

        return self.final_outputs
