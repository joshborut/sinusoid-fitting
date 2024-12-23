# Note: Portions of this code were generated with ChatGPT
import numpy as np
from dataset_generator import get_dataset, scatter_plot_two_datasets
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


# def loss(params, examples, labels):
#     predictions = forward_pass(params, jnp.reshape(jnp.array([examples]), (len(labels), 1)))
#     return mean_squared_error(labels, predictions)


def shuffle_dataset(examples, labels):
    indices = np.random.permutation(len(examples))
    return examples[indices], labels[indices]


def get_average_dataset_deviation(predictions, labels):
    return np.mean(np.abs(predictions - labels))


def training_loop(neural_net, train_examples, train_labels, test_examples, test_labels, epochs, minibatch_size=20):
    optimizer = optim.SGD(neural_net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        train_examples, train_labels = shuffle_dataset(train_examples, train_labels)
        starting_indices = range(0, len(train_examples), minibatch_size)
        for starting_index in starting_indices:
            minibatch_examples = train_examples[
                                 starting_index: min(starting_index + minibatch_size, len(train_examples))].reshape(
                (32, 1))
            minibatch_examples = torch.from_numpy(minibatch_examples).to(torch.float32).reshape((minibatch_size, 1))

            minibatch_labels = train_labels[starting_index: min(starting_index + minibatch_size, len(train_labels))]
            minibatch_labels = torch.from_numpy(minibatch_labels).to(torch.float32).reshape((minibatch_size, 1))

            minibatch_predictions = neural_net(minibatch_examples)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(minibatch_predictions, minibatch_labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch #{epoch + 1}")

        # Assess on test set
        predicted_test_labels = neural_net(test_examples)
        print(get_average_dataset_deviation(predicted_test_labels.detach().numpy(), test_labels))


#
#     # Graph test results
#     scatter_plot_two_datasets(test_examples, test_labels, test_examples, predicted_test_labels, "True", "Predicted",
#                               "NN Performance")


if __name__ == '__main__':
    dataset_size = 10000
    examples, labels = get_dataset(3, 2, dataset_size)

    training_set_ratio = 0.8
    training_set_size = int(dataset_size * training_set_ratio)
    train_examples = examples[:training_set_size]
    train_labels = labels[:training_set_size]
    test_examples = torch.from_numpy(examples[training_set_size:]).to(torch.float32).reshape((2000, 1))
    test_labels = labels[training_set_size:]

    neural_net = NeuralNetwork([1, 40, 40, 1])
    output = neural_net(test_examples)

    # Network setup
    layer_sizes = [1, 40, 40, 1]  # Input layer (1), two hidden layers (40 each), output layer (1)
    # key = jrand.PRNGKey(0)

    # Initialize parameters
    training_loop(neural_net, train_examples, train_labels, test_examples, test_labels, 5, 32)
