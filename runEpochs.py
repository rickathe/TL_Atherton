
import numpy as np
import matplotlib.pyplot as plt
import nn4layer

def plot_loss(test_0_output, test_1_output, test_2_output, test_3_output, 
    hyper_p, test_runs, test_name):
    """ Creates a plot to chart error over time. Takes 4 inputs. """
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
    plt.savefig(f"plots/{test_name}.png")
    plt.show()

def histogram_error(val_mean, val_std, test_runs, hyper_p, layers, test_name,
    iters, epochs_final):
    count_hist = 0
    labels = test_runs
    x_pos = np.arange(len(labels))
    val_means = list(val_mean)
    val_stds = list(val_std)
    fig, ax = plt.subplots()
    bars = ax.bar(x_pos, val_means, yerr=val_stds, align='center', alpha=0.5, 
        ecolor='black', capsize=10)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, round(yval,1))
        plt.text(bar.get_x(), 50, epochs_final[count_hist])
        count_hist += 1
    ax.set_ylabel('Mean Error')
    ax.set_xlabel(f'{hyper_p}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(f'{layers} Mean Min. Error ({iters} Trials per Bar)')
    ax.yaxis.grid(True)
    plt.tight_layout
    plt.savefig(f'plots/{test_name}.png')
    plt.show()


# Run the neural network.
set_input_nodes = 2
set_hidden_nodes = 20
set_output_nodes = 1
set_learning_rate = 0.01

training_size = 10000
testing_size = 1000
epoch = 1000

# Number of times to train each hyper-param. Results will be averaged.
iters = 5

# Allows for plotting either training over epochs or a bar graph with minimum 
# mean error. use 'bar' for bar graph and 'plot' for training line graph.
test_type = 'bar'
#test_type = 'plot'

# Select hyperparameter to be analyzed and values to cycle through.
# NOTE: Where these iteract with the NN are currently hard-coded and must be
# changed manually.
test_runs = [8, 16, 32, 64]
hyper_p = "HN"a
test_name = "plot_10k1k_var_01lr_4layer_test3"
layers = "4 Hidden Layer"

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
validation_graph = []
epoch_broken = []
epochs_final = []


for test in range(len(test_runs)):
    # Controls the hyper-parameters to be tested, set by var test_runs.

    for rounds in range(iters):
        # Controls the number of times to test each hyper-parameter.

        # Change this nn#layer when changing the number of hidden layers to 
        # use.
        n = nn4layer.NeuralNetwork(set_input_nodes, test_runs[test], 
            set_output_nodes, set_learning_rate)

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


            for record in range(testing_size):
                # Checks the trained network against the validation data and 
                # records the difference.
                output = n.query(scaled_testing_data[record, :])
                output = output * (data_max - data_min) + data_min

                # Records the magnitude of the difference between actual and 
                # prediction.
                validation_loss += abs(testing_solutions[0, record] - output)
            
            validation_loss = validation_loss / testing_size
            validation_record.append(validation_loss)

            if e > 100 and validation_record[-1] > validation_record[-2]:
                
                epoch_broken.append(e)
                
                if (rounds + 1) % iters is 0:
                    # Before the last iteration of a hyper-param test breaks,
                    # network records the average epochs it took to minimize
                    # error.
                    epochs_final.append(sum(epoch_broken)/len(epoch_broken))
                    epoch_broken = []
                break

        if test_type is 'plot':

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
            elif test is 4:
                validation_record_4 = validation_record

        elif test_type is 'bar':

            if rounds is 0:
                validation_final = np.array(validation_record[-2])
            else:
                validation_final = np.append(validation_final, 
                    validation_record[-2])
        
        validation_record = []
    
    if test_type is 'bar':

        validation_graph.append(validation_final) 


if test_type is 'plot':

    # Flatten list of arrays from for loops into something the graph can 
    # utilize.
    #record_final = np.concatenate(training_record)
    #validation_final = np.concatenate(validation_record)
    test_0_output = np.concatenate(validation_record_0)

    test_1_output = np.concatenate(validation_record_1)
    test_2_output = np.concatenate(validation_record_2)
    test_3_output = np.concatenate(validation_record_3)
    test_4_output = np.concatenate(validation_record_4)

    # Plot the graph.
    plot_loss(test_0_output, test_1_output, test_2_output, test_3_output,
        hyper_p, test_runs, test_name)

if test_type is 'bar':

    validation_mean = np.mean(validation_graph, axis=1)
    validation_std = np.std(validation_graph, axis=1)

    histogram_error(validation_mean, validation_std, test_runs, hyper_p, 
        layers, test_name, iters, epochs_final)



