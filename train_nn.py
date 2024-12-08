# Note: Portions of this code were generated with ChatGPT
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrand, jit, grad

from dataset_generator import get_dataset, scatter_plot_two_datasets


def mean_squared_error(labels, predictions):
    return 0.5 * jnp.mean((labels - predictions) ** 2)


def loss(params, examples, labels):
    predictions = forward_pass(params, jnp.reshape(jnp.array([examples]), (len(labels), 1)))
    return mean_squared_error(labels, predictions)


def shuffle_dataset(examples, labels):
    indices = np.random.permutation(len(examples))
    return examples[indices], labels[indices]


def get_average_dataset_deviation(predictions, labels):
    return np.mean(np.abs(predictions - labels))


def initialize_params(layer_sizes, key):
    """
    Initialize parameters for a fully connected neural network.

    Parameters:
    layer_sizes (list): List of integers specifying the size of each layer.
    key (jax.rand.PRNGKey): Random key for parameter initialization.

    Returns:
    params (list): List of tuples containing (weights, biases) for each layer.
    """
    params = []
    keys = jrand.split(key, len(layer_sizes) - 1)

    for in_size, out_size, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        weight_key, bias_key = jrand.split(k)
        weights = jrand.normal(weight_key, (in_size, out_size)) * jnp.sqrt(2.0 / in_size)
        biases = jnp.zeros(out_size)
        params.append((weights, biases))

    return params


def forward_pass(params, x):
    """
    Perform a forward pass through the network.

    Parameters:
    params (list): Network parameters, a list of (weights, biases) tuples.
    x (jax.numpy.array): Input data.

    Returns:
    jax.numpy.array: Output of the network.
    """
    for i, (weights, biases) in enumerate(params[:-1]):
        x = jnp.dot(x, weights) + biases
        x = jax.nn.relu(x)  # Apply ReLU activation for hidden layers

    # Final layer (linear output)
    final_weights, final_biases = params[-1]
    output = jnp.dot(x, final_weights) + final_biases
    return output


@jit
def update_params(params, x, y):
    lr = 0.001
    """
    Perform one step of gradient descent to update parameters.

    Parameters:
    params (list): Network parameters.
    x (jax.numpy.array): Input data.
    y (jax.numpy.array): Ground truth target data.
    lr (float): Learning rate.

    Returns:
    list: Updated network parameters.
    """
    grads = grad(loss)(params, x, y)  # Compute gradients of the loss w.r.t. parameters
    updated_params = [
        (w - lr * dw, b - lr * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]
    return updated_params


def training_loop(params, train_examples, train_labels, test_examples, test_labels, epochs, minibatch_size=20):
    for epoch in range(epochs):
        train_examples, train_labels = shuffle_dataset(train_examples, train_labels)
        starting_indices = range(0, len(train_examples), minibatch_size)
        for starting_index in starting_indices:
            minibatch_examples = train_examples[
                                 starting_index: min(starting_index + minibatch_size, len(train_examples))]
            minibatch_labels = train_labels[starting_index: min(starting_index + minibatch_size, len(train_labels))]
            params = update_params(params, minibatch_examples, minibatch_labels)

        print(f"Epoch #{epoch + 1}")

        # Assess on test set
        predicted_test_labels = forward_pass(params, jnp.reshape(jnp.array([test_examples]), (len(test_labels), 1)))
        print(get_average_dataset_deviation(jnp.squeeze(predicted_test_labels), test_labels))

    # Graph test results
    scatter_plot_two_datasets(test_examples, test_labels, test_examples, predicted_test_labels, "True", "Predicted",
                              "NN Performance")


if __name__ == '__main__':
    dataset_size = 10000
    examples, labels = get_dataset(3, 2, dataset_size)

    training_set_ratio = 0.8
    training_set_size = int(dataset_size * training_set_ratio)
    train_examples = examples[:training_set_size]
    train_labels = labels[:training_set_size]
    test_examples = examples[training_set_size:]
    test_labels = labels[training_set_size:]

    # Network setup
    layer_sizes = [1, 40, 40, 1]  # Input layer (1), two hidden layers (40 each), output layer (1)
    key = jrand.PRNGKey(0)

    # Initialize parameters
    params = initialize_params(layer_sizes, key)
    training_loop(params, train_examples, train_labels, test_examples, test_labels, 5, 32)
