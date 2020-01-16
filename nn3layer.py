import numpy as np


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, 
	learning_rate):
        """ Initializes the values of the network """
        
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate



        # Initializes weights with a normal distribution with a mean of 0, 
		# an std of inodes^(-0.5), and a shape of hnodes x inodes.
        self.wih1 = np.random.normal(0.0, pow(self.inodes, -0.5), \
			(self.hnodes, self.inodes))

        self.wh1h2 = np.random.normal(0.0, pow(self.hnodes, -0.5), \
            (self.hnodes, self.hnodes))
        self.wh2h3 = np.random.normal(0.0, pow(self.hnodes, -0.5), \
            (self.hnodes, self.hnodes))
        self.wh3o = np.random.normal(0.0, pow(self.hnodes, -0.5), \
			(self.onodes, self.hnodes))
        
        # Anon function to call sigmoid function without importing scipy.
        self.sigmoid = lambda x : (1 / (1 + np.exp(-x)))


    def train(self, inputs_list, targets_list):
        """ Takes the inputs and target values for a network and forward
        propagates the data. It then compares target value with actual value
        and propagates the errors back through the nodes, updating weights.
        """

        # Creates 2-dimensional arrays from lists. Will prepend 1s to the 
		# array if not at least ndim=2.
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
		# Calcs dot product of input-hidden weights and input values, outputs
		# an array with summed input values for each hnode.
        hidden1_inputs = np.dot(self.wih1, inputs)
		# Applies activation function to summed inputs to a hnode. Can be 
		# eventually combined with above line.
        hidden1_outputs = self.sigmoid(hidden1_inputs)

        hidden2_inputs = np.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.sigmoid(hidden2_inputs)
        hidden3_inputs = np.dot(self.wh2h3, hidden2_outputs)
        hidden3_outputs = self.sigmoid(hidden3_inputs)
		# Calcs dot product of hidden-output weights and input values, outputs
		# an array w/ summed output values for each onode. Applies sigmoid.
        final_inputs = np.dot(self.wh3o, hidden3_outputs)
        final_outputs = self.sigmoid(final_inputs)
		# Calcs error between real answer and NN prediction.
        output_errors = targets - final_outputs
		# Proportionally calcs how much correct each hnode receives.
        hidden3_errors = np.dot(self.wh3o.T, output_errors)
        hidden2_errors = np.dot(self.wh2h3.T, hidden3_errors)
        hidden1_errors = np.dot(self.wh1h2.T, hidden2_errors)
		# Updates hidden-output and input-hidden weights based on error.
        self.wh3o += self.lr * np.dot((output_errors * final_outputs * \
			(1.0 - final_outputs)), np.transpose(hidden3_outputs))
        self.wh2h3 += self.lr * np.dot((hidden3_errors * hidden3_outputs * \
			(1.0 - hidden3_outputs)), np.transpose(hidden2_outputs)) 
        self.wh1h2 += self.lr * np.dot((hidden2_errors * hidden2_outputs * \
			(1.0 - hidden2_outputs)), np.transpose(hidden1_outputs))            
        self.wih1 += self.lr * np.dot((hidden1_errors * hidden1_outputs * \
			(1.0 - hidden1_outputs)), np.transpose(inputs))
          
        return output_errors


    def query(self, inputs_list):
        """ Forward propagates validation data through the network to check
        performance. Check train function above for comments on how commands
        below function.
        """

        inputs = np.array(inputs_list, ndmin=2).T
        hidden1_inputs = np.dot(self.wih1, inputs)
        hidden1_outputs = self.sigmoid(hidden1_inputs)
        hidden2_inputs = np.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.sigmoid(hidden2_inputs)
        hidden3_inputs = np.dot(self.wh2h3, hidden2_outputs)
        hidden3_outputs = self.sigmoid(hidden3_inputs)
        final_inputs = np.dot(self.wh3o, hidden3_outputs)
        final_outputs = self.sigmoid(final_inputs)
        return final_outputs