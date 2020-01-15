import numpy as np
import matplotlib.pyplot as plt


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
        self.wh3h4 = np.random.normal(0.0, pow(self.hnodes, -0.5), \
            (self.hnodes, self.hnodes))
        self.wh4h5 = np.random.normal(0.0, pow(self.hnodes, -0.5), \
            (self.hnodes, self.hnodes))        
        self.wh5o = np.random.normal(0.0, pow(self.hnodes, -0.5), \
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
        hidden4_inputs = np.dot(self.wh3h4, hidden3_outputs)
        hidden4_outputs = self.sigmoid(hidden4_inputs)
        hidden5_inputs = np.dot(self.wh4h5, hidden4_outputs)
        hidden5_outputs = self.sigmoid(hidden5_inputs)
		# Calcs dot product of hidden-output weights and input values, outputs
		# an array w/ summed output values for each onode. Applies sigmoid.
        final_inputs = np.dot(self.wh5o, hidden5_outputs)
        final_outputs = self.sigmoid(final_inputs)
		# Calcs error between real answer and NN prediction.
        output_errors = targets - final_outputs
		# Proportionally calcs how much correct each hnode receives.
        hidden5_errors = np.dot(self.wh5o.T, output_errors)
        hidden4_errors = np.dot(self.wh4h5.T, hidden5_errors)
        hidden3_errors = np.dot(self.wh3h4.T, hidden4_errors)
        hidden2_errors = np.dot(self.wh2h3.T, hidden3_errors)
        hidden1_errors = np.dot(self.wh1h2.T, hidden2_errors)
		# Updates hidden-output and input-hidden weights based on error.
        self.wh5o += self.lr * np.dot((output_errors * final_outputs * \
			(1.0 - final_outputs)), np.transpose(hidden5_outputs))
        self.wh4h5 += self.lr * np.dot((hidden5_errors * hidden5_outputs * \
			(1.0 - hidden5_outputs)), np.transpose(hidden4_outputs)) 
        self.wh3h4 += self.lr * np.dot((hidden4_errors * hidden4_outputs * \
			(1.0 - hidden4_outputs)), np.transpose(hidden3_outputs)) 
        self.wh2h3 += self.lr * np.dot((hidden3_errors * hidden3_outputs * \
			(1.0 - hidden3_outputs)), np.transpose(hidden2_outputs)) 
        self.wh1h2 += self.lr * np.dot((hidden2_errors * hidden2_outputs * \
			(1.0 - hidden2_outputs)), np.transpose(hidden1_outputs))            
        self.wih1 += self.lr * np.dot((hidden1_errors * hidden1_outputs * \
			(1.0 - hidden1_outputs)), np.transpose(inputs))
       
        # Removes scaling from errors.
        errors = output_errors * (data_max - data_min) + data_min
       
        return errors


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
        hidden4_inputs = np.dot(self.wh3h4, hidden3_outputs)
        hidden4_outputs = self.sigmoid(hidden4_inputs)
        hidden5_inputs = np.dot(self.wh4h5, hidden4_outputs)
        hidden5_outputs = self.sigmoid(hidden5_inputs)
        final_inputs = np.dot(self.wh5o, hidden5_outputs)
        final_outputs = self.sigmoid(final_inputs)
        return final_outputs


# Creates testing data to train the network to add.
training_size = 10000
# Creates m rows of 2 integer values to act as inputs.
training_data = np.random.randint(1, 100, (training_size, 2))
# Multiplies the 2 initial values together to get a real answer, sets to a
# m x 1 array.
training_solutions = np.array(np.multiply(training_data[:,0], \
    training_data[:,1]), ndmin=2)

# Creates validation data to test the network.
testing_size = 1000
testing_data = np.random.randint(1, 100, (testing_size,2))
testing_solutions = np.array(np.multiply(testing_data[:,0], \
    testing_data[:,1]), ndmin=2)

# Takes the maxes and mins of training and testing arrays.
data_max = np.maximum(np.max(training_solutions), np.max(testing_solutions))
data_min = np.minimum(np.min(training_data), np.min(testing_data))

# Scales training and validation data to be 0 < x <= 1.
scaled_training_data = (training_data - data_min) / (data_max - data_min)
scaled_training_solutions = (training_solutions - data_min) \
    / (data_max - data_min)

scaled_testing_data = (testing_data - data_min) / (data_max - data_min)
scaled_testing_solutions = (testing_solutions - data_min) \
    / (data_max - data_min)

# Run the neural network.
set_input_nodes = 2
set_hidden_nodes = 50
set_output_nodes = 1
set_learning_rate = 0.001

n = NeuralNetwork(set_input_nodes, set_hidden_nodes, set_output_nodes, \
    set_learning_rate)

# Lists for plotting loss. Records errors from training and validation.
training_record = []
validation_record = []

validation_accuracy_10 = []
validation_accuracy_5 = []
validation_accuracy_2 = []
validation_accuracy_1 = []
finish = 0

# Number of epochs to train the data.
epoch = 5000

for e in range(epoch):
    loss = 0
    validation_loss = 0
    for record in range(training_size):
        error = n.train(scaled_training_data[record, :], \
            scaled_training_solutions[0, record])
        loss += abs(error)
    loss = loss / training_size
    training_record.append(loss)

    # Checks the trained network against the validation data and records the
    # difference.
    for record in range(testing_size):
        output = n.query(scaled_testing_data[record, :])
        output = output * (data_max - data_min) + data_min

        # Records the magnitude of the difference between actual and 
        # prediction.
        validation_loss += abs(testing_solutions[0, record] - output)
        
        # Records the accuracy within 10%/5%/1% for the last epoch.
        if finish is 1:
            solution_min_10 = testing_solutions[0, record] * .9
            solution_max_10 = testing_solutions[0, record] * 1.1
            solution_min_5 = testing_solutions[0, record] * .95
            solution_max_5 = testing_solutions[0, record] * 1.05
            solution_min_2 = testing_solutions[0, record] * .98
            solution_max_2 = testing_solutions[0, record] * 1.02
            solution_min_1 = testing_solutions[0, record] * .99
            solution_max_1 = testing_solutions[0, record] * 1.01
            if solution_min_10 < output < solution_max_10:
                validation_accuracy_10.append(output)
            if solution_min_5 < output < solution_max_5:
                validation_accuracy_5.append(output)
            if solution_min_2 < output < solution_max_2:
                validation_accuracy_2.append(output)
            if solution_min_1 < output < solution_max_1:
                validation_accuracy_1.append(output)
    
    validation_loss = validation_loss / testing_size
    validation_record.append(validation_loss)

    # Breaks if the loss of the present epoch is larger then the last epoch.
    if finish is 1:
        break
    if e > 300:
        if validation_record[-1] > validation_record[-2]:
            finish = 1


def plot_loss(training_errors, validation_errors):
    """ Creates a plot to chart error over time. """
    plt.xscale('Log')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Actual Error')
    plt.plot(training_errors, label = "Validation Error", \
        color = 'red')
    plt.plot(validation_errors, label = "Training Error", \
        color = 'blue')
    plt.legend()
    # Saves plot automatically, adjust filename as needed.
    plt.savefig('multiply_10k_1k_50h_001lr_5layer_test6.png')
    plt.show()

# Flatten list of arrays from for loops into something the graph can utilize.
record_final = np.concatenate(training_record)
validation_final = np.concatenate(validation_record)

# Print 10%/5%/2%/1% accuracy percentages.
accuracy_10 = (len(validation_accuracy_10) / testing_solutions.shape[1]) * 100
accuracy_5 = (len(validation_accuracy_5) / testing_solutions.shape[1]) * 100
print("Validation accuracy within 10% is:", accuracy_10, "%")
print("Validation accuracy within 5% is is:", accuracy_5, "%")
accuracy_2 = (len(validation_accuracy_2) / testing_solutions.shape[1]) * 100
accuracy_1 = (len(validation_accuracy_1) / testing_solutions.shape[1]) * 100
print("Validation accuracy within 2% is:", accuracy_2, "%")
print("Validation accuracy within 1% is is:", accuracy_1, "%")

# Show off some data.
show_data = np.random.randint(0, 100, (15,2))
show_solutions = np.array(np.multiply(show_data[:,0], \
    show_data[:,1]), ndmin=2)
scaled_show_data = (show_data - np.min(show_data)) / (np.max(show_solutions) - np.min(show_data))
scaled_show_solutions = (show_solutions.T - np.min(show_data)) \
    / (np.max(show_solutions) - np.min(show_data))
for record in range(15):
    output = n.query(scaled_show_data[record, :])
    output = output * (np.max(show_solutions) - np.min(show_data)) + np.min(show_data)
    print(show_data[record,0], "times", show_data[record,1], "is", output, ".. actual is:", (show_data[record, 0] * show_data[record, 1]))

# Plot the graph.
plot_loss(record_final, validation_final)