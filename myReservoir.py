import numpy as np
import matplotlib.pyplot as plt
import reservoir2

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
    plt.savefig('reservoir_RA_10whdens_100h_7spec_test_2.png')
    plt.show()


# Run the neural network.
set_input_nodes = 2
set_hidden_nodes = 100
set_output_nodes = 1
set_learning_rate = 0.01
spectral_radius = 0.7
connectivity = 0.5

training_size = 10000
testing_size = 1000
epoch = 1000

n = reservoir2.Reservoir(set_input_nodes, set_hidden_nodes, set_output_nodes, \
    set_learning_rate, spectral_radius, connectivity)


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

# Lists for plotting loss. Records errors from training and validation.
training_record = []
validation_record = []

validation_accuracy_10 = []
validation_accuracy_5 = []
validation_accuracy_2 = []
validation_accuracy_1 = []
finish = 0

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
    '''
    # Breaks if the loss of the present epoch is larger then the last epoch.
    if finish is 1:
        break
    if e > 300:
        if validation_record[-1] > validation_record[-2]:
            finish = 1
    '''
# Flatten list of arrays from for loops into something the graph can utilize.
training_final = np.concatenate(training_record)
validation_final = np.concatenate(validation_record)

# Print 10%/5%/2%/1% accuracy percentages.
accuracy_10 = (len(validation_accuracy_10) / testing_size) * 100
accuracy_5 = (len(validation_accuracy_5) / testing_size) * 100
accuracy_2 = (len(validation_accuracy_2) / testing_size) * 100
accuracy_1 = (len(validation_accuracy_1) / testing_size) * 100
print("Validation accuracy within 10% is:", accuracy_10, "%")
print("Validation accuracy within 5% is is:", accuracy_5, "%")
print("Validation accuracy within 2% is:", accuracy_2, "%")
print("Validation accuracy within 1% is is:", accuracy_1, "%")


# Show off some data.
show_data = np.random.randint(0, 100, (15,2))
show_solutions = np.array(np.multiply(show_data[:,0], \
    show_data[:,1]), ndmin=2)
scaled_show_data = (show_data - np.min(show_data)) / (np.max(show_solutions) \
    - np.min(show_data))
scaled_show_solutions = (show_solutions.T - np.min(show_data)) \
    / (np.max(show_solutions) - np.min(show_data))
for record in range(15):
    output = n.query(scaled_show_data[record, :])
    output = output * (np.max(show_solutions) - np.min(show_data)) \
        + np.min(show_data)
    print(show_data[record,0], "times", show_data[record,1], "is", \
        output, ".. actual is:", (show_data[record, 0] * show_data[record, 1]))


# Plot the graph.
plot_loss(training_final, validation_final)
