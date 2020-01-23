
import numpy as np
import matplotlib.pyplot as plt
import nn5layer

def plot_loss(test_0_output, test_1_output, test_2_output, test_3_output):
    """ Creates a plot to chart error over time. """
    plt.xscale('Log')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Actual Error')
    plt.plot(test_0_output, label = f"{hyper_p} {test_runs[0]} Error", 
        color = 'red')
    plt.plot(test_1_output, label = f"{hyper_p} {test_runs[1]} Error", 
        color = 'blue')
    plt.plot(test_2_output, label = f"{hyper_p} {test_runs[2]} Error", 
        color = 'green')
    plt.plot(test_3_output, label = f"{hyper_p} {test_runs[3]} Error", 
        color = 'orange')
    plt.plot(test_3_output, label = f"{hyper_p} {test_runs[4]} Error", 
        color = 'cyan')
    plt.title(f"{test_name}")
    plt.legend()
    # Saves plot automatically, adjust filename as needed.
    plt.savefig(f"plots/{test_name}")
    plt.show()


# Run the neural network.
set_input_nodes = 2
set_hidden_nodes = 20
set_output_nodes = 1
set_learning_rate = 0.01

training_size = 10000
testing_size = 1000
epoch = 3000

# Select hyperparameter to be analyzed and values to cycle through. 
# NOTE: Where these iteract with the NN are currently hard-coded and must be
# changed manually.
test_runs = [2, 4, 8, 16, 32]
hyper_p = "HN"
test_name = "multiply_10k1k_var_01lr_4layer_test1"

# Creates m rows of 2 integer values to act as inputs.
training_data = np.random.randint(1, 100, (training_size, 2))
# Multiplies the 2 initial values together to get a real answer, sets to a
# m x 1 array.
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

# Lists for plotting loss. Records errors from training and validation.
training_record = []
validation_record = []


for test in range(len(test_runs)):
    n = nn5layer.NeuralNetwork(set_input_nodes, test_runs[test], set_output_nodes, \
    set_learning_rate)
    for e in range(epoch):
        loss = 0
        validation_loss = 0
        for record in range(training_size):
            error = n.train(scaled_training_data[record, :], \
                scaled_training_solutions[0, record])
            error = error * (data_max - data_min) + data_min
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
        
        validation_loss = validation_loss / testing_size
        validation_record.append(validation_loss)

        if e > 300 and validation_record[-1] > validation_record[-2]:
            break

    if test is 0:
        #test_data = np.asarray(validation_record)
        validation_record_0 = validation_record
    elif test is 1:
        #test_data_temp = np.asarray(validation_record)
        #np.vstack((test_data, test_data_temp))
        validation_record_1 = validation_record
    elif test is 2:
        validation_record_2 = validation_record
    elif test is 3:
        validation_record_3 = validation_record
    else:
        validation_record_4 = validation_record
    validation_record = []

# Flatten list of arrays from for loops into something the graph can utilize.
#record_final = np.concatenate(training_record)
#validation_final = np.concatenate(validation_record)
test_0_output = np.concatenate(validation_record_0)
test_1_output = np.concatenate(validation_record_1)
test_2_output = np.concatenate(validation_record_2)
test_3_output = np.concatenate(validation_record_3)
test_4_output = np.concatenate(validation_record_4)


# Plot the graph.
'''
np.savetxt(f'testData/{test_name}_data0.txt', test_0_output, delimiter=',')
np.savetxt(f'testData/{test_name}_data1.txt', test_1_output, delimiter=',') 
np.savetxt(f'testData/{test_name}_data2.txt', test_2_output, delimiter=',') 
np.savetxt(f'testData/{test_name}_data3.txt', test_3_output, delimiter=',')
'''
plot_loss(test_0_output, test_1_output, test_2_output, test_3_output)
