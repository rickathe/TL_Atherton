import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


def plot_loss(training_errors, validation_errors):
    """ Creates a plot to chart error over time. """
    plt.xscale('Log')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Actual Error')
    plt.plot(training_errors, label = "Training Error", \
        color = 'blue')
    plt.plot(validation_errors, label = "Validation Error", \
        color = 'red')
    plt.legend()
    # Saves plot automatically, adjust filename as needed.
    plt.savefig('reservoir_uniform_input_oscillation.png')
    plt.show()


class Reservoir:

    def __init__(self, inodes, rnodes, onodes, learning_rate, sparsity):

        self.inodes = inodes
        self.rnodes = rnodes
        self.onodes = onodes
        self.lr = learning_rate

        # Initialize weights
        self.Win = np.random.uniform(0.0,1.0, (self.rnodes, self.inodes))
        self.Wh = sparse.rand(self.rnodes, self.rnodes, 
            density = .0025).todense()
        self.Wback = np.random.normal(0.0, pow(self.rnodes, -0.5), 
            (self.onodes, (self.rnodes)))
        self.Wout = np.random.uniform(0.0, 1.0, (self.onodes, self.onodes
            + self.rnodes + self.inodes))
        
        self.Wh[np.where(self.Wh > 0)] -= 0.5

        # Reservoir activation function
        self.relu = lambda x : np.maximum(x, 0)

        # Apply spectral radius of 0.9 on res-res weights
        ## Check this -- read up on eigenvalues
        spec_rad = max(abs(np.linalg.eig(self.Wh)[0]))
        self.Wh /= spec_rad / 0.9
        
    def compute_state(self, inputs, targets):
        
        # N = training examples, T = variables in each example
        N, T = inputs.shape

        inputs = np.array(inputs, ndmin=2)
        targets = np.array(targets, ndmin=2).T
        
        # Init state for reservoir
        previous_state = np.zeros((self.rnodes, 1), dtype=float)
        previous_output = np.zeros((1,1))
        #state_matrix = np.empty((N, T, self.rnodes), dtype=float)
        
        # Init for record of mean error per epoch 
        training_record = np.zeros((N), dtype=float)

        for n in range(N):
            
            current_input = inputs[n]
            current_target = targets[n]


            wh_part = self.Wh @ previous_state

            # Needs to be reshaped or it cannot be broadcast
            win_part = self.Win @ np.reshape(current_input.T,(2,1))
            wback_part = self.Wback.T @ previous_output
            #print("Inputs are", inputs[n])
            #print("Output is", targets[n])

            #wout_part = self.Wout.T @ previous_output
            
            current_state = wh_part + win_part + wback_part
            current_state = np.tanh(current_state)
            
            # Smaller array is broadcast across the larger array
            #current_state = np.tanh(self.Wh @ previous_state.T + self.Win
            #    @ current_input.T + self.Wout.T * previous_output).T
            
            # Stores previous state [N x reservoir nodes] into state matrix
            #state_matrix[:, n] = current_state
            current_output = np.concatenate([current_state, np.reshape(current_input.T,(2,1)),
                previous_output])
            current_output = np.tanh(self.Wout @ current_output)

            output_error = current_target - current_output

            self.Wback += self.lr * np.dot((output_error * (1.0 - np.square(current_output))), current_state.T)

            previous_state = current_state
            previous_output = current_output

            training_record[n] = output_error

        return np.mean(training_record)


inodes = 2
rnodes = 1000
onodes = 1
learning_rate = 0.01
sparsity = 0.05
epochs = 1000
training_size = 10000
testing_size = 1000

# Creates m rows of 2 integer values to act as inputs.
training_data = np.random.randint(1, 100, (training_size, 2))
# Multiplies the 2 initial values together to get a real answer, sets to a
# m x 1 array.
#training_solutions = np.array(np.multiply(training_data[:,0], \
#    training_data[:,1]), ndmin=2)
training_solutions = np.array(np.multiply(training_data[:,0], \
    training_data[:,1]), ndmin=2)

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



R = Reservoir(inodes, rnodes, onodes, learning_rate, sparsity)

training_error = np.zeros(epochs)

for e in range(epochs):
    
    training_error[e] = R.compute_state(scaled_training_data, scaled_training_solutions)

#training_error = training_error * (data_max - data_min) + data_min

plot_loss(training_error, np.zeros(epochs))
