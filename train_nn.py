# Note: Portions of this code were generated with ChatGPT
import numpy as np
from dataset_generator import get_train_and_test_data, scatter_plot_two_datasets, scatter_plot, \
    save_scatter_plot_two_datasets
import torch
from torch import nn
import torch.optim as optim
from collections import OrderedDict


class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(NeuralNetwork, self).__init__()

        layers = OrderedDict()
        for layer_num in range(len(layer_sizes) - 1):
            layers[f"lin{layer_num + 1}"] = nn.Linear(layer_sizes[layer_num], layer_sizes[layer_num + 1])

            if layer_num != len(layer_sizes) - 2:
                layers[f"relu{layer_num + 1}"] = nn.ReLU()

        self.layers = nn.Sequential(layers)

    def forward(self, x):
        out = self.layers(x)
        return out


def shuffle_dataset(examples, labels):
    """Mix the order of pairs of examples and labels.

    :param examples: List of examples
    :param labels: List of labels
    :return: Shuffled lists of examples and labels
    """
    indices = np.random.permutation(len(examples))
    return examples[indices], labels[indices]


def get_average_dataset_deviation(predictions, labels):
    """Get the average error between two lists.

    :param predictions: List of predictions
    :param labels: List of labels
    :return: Average error
    """
    predictions = np.squeeze(predictions)
    pred_error = predictions - labels
    pred_error_mag = np.abs(pred_error)
    mean_pred_error_mag = np.mean(pred_error_mag)
    return mean_pred_error_mag


def plot_predictions_and_labels(examples, labels, neural_net, title="Predictions vs examples"):
    """Plot the neural net predictions for labels.

    :param examples: List of examples
    :param labels: List of labels
    :param neural_net: Neural net object
    :param title: Plot title
    :return: None
    """
    with torch.no_grad():
        predictions = neural_net(examples)
        torch_labels = torch.from_numpy(labels).to(torch.float32).reshape(predictions.shape)
        loss = torch.nn.functional.mse_loss(predictions, torch_labels)
        print(f"Loss: {loss}")
        print(get_average_dataset_deviation(predictions.numpy(), labels))
    scatter_plot_two_datasets(examples, labels, examples, predictions, "Labels", "Predictions",
                              title=title)


def save_predictions_and_labels(examples, labels, neural_net, title="Predictions vs examples"):
    """Plot the neural net predictions for labels.

    :param examples: List of examples
    :param labels: List of labels
    :param neural_net: Neural net object
    :param title: Plot title
    :return: None
    """
    with torch.no_grad():
        predictions = neural_net(examples)
        torch_labels = torch.from_numpy(labels).to(torch.float32).reshape(predictions.shape)
        loss = torch.nn.functional.mse_loss(predictions, torch_labels)
        print(f"Loss: {loss}")
        print(get_average_dataset_deviation(predictions.numpy(), labels))
    save_scatter_plot_two_datasets(examples, labels, examples, predictions, "Labels", "Predictions",
                                   title=title)


def training_loop(neural_net, train_examples, train_labels, test_examples, test_labels, epochs, minibatch_size=20):
    """Train the neural net.

    :param neural_net: Neural net object
    :param train_examples: List of training examples
    :param train_labels: List of training labels
    :param test_examples: List of test examples
    :param test_labels: List of test labels
    :param epochs: Number of epochs
    :param minibatch_size: Size of minibatch
    :return: None
    """
    optimizer = optim.SGD(neural_net.parameters(), lr=0.001, momentum=0.9)

    save_predictions_and_labels(test_examples, test_labels, neural_net,
                                title=f"Predictions vs labels before training")

    for epoch_num in range(epochs):
        train_examples, train_labels = shuffle_dataset(train_examples, train_labels)
        starting_indices = range(0, len(train_examples), minibatch_size)

        for starting_index in starting_indices:
            minibatch_examples = train_examples[
                                 starting_index: min(starting_index + minibatch_size, len(train_examples))]
            minibatch_examples = torch.from_numpy(minibatch_examples).to(torch.float32).reshape((minibatch_size, 1))

            minibatch_labels = train_labels[
                               starting_index: min(starting_index + minibatch_size, len(train_labels))]
            minibatch_labels = torch.from_numpy(minibatch_labels).to(torch.float32).reshape((minibatch_size, 1))

            minibatch_predictions = neural_net(minibatch_examples)
            loss = torch.nn.functional.mse_loss(minibatch_predictions, minibatch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Plot epoch outcome
        print(f"Epoch #{epoch_num + 1} loss: {loss.item()}")
        save_predictions_and_labels(test_examples, test_labels, neural_net,
                                    title=f"Predictions vs labels (Epoch #{epoch_num + 1})")

    # Assess on test set
    with torch.no_grad():
        predicted_test_labels = neural_net(test_examples)
        print(get_average_dataset_deviation(predicted_test_labels.detach().numpy(), test_labels))

    # Plot final results
    scatter_plot_two_datasets(test_examples, test_labels, test_examples, predicted_test_labels, "True", "Predicted",
                              title="NN Performance")


if __name__ == '__main__':
    # Generate the dataset
    dataset_size = 10000
    training_set_ratio = 0.8
    train_examples, train_labels, test_examples, test_labels = get_train_and_test_data(dataset_size, training_set_ratio,
                                                                                       3, 2)

    # Initialize the neural net
    neural_net = NeuralNetwork([1, 40, 40, 1])  # Input layer (1), two hidden layers (40 each), output layer (1)

    # Network setup
    layer_sizes = [1, 40, 40, 1]  # Input layer (1), two hidden layers (40 each), output layer (1)

    # Train the neural net
    training_loop(neural_net, train_examples, train_labels, test_examples, test_labels, 5, 32)
