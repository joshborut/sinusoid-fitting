# Note: Portions of this code were generated using ChatGPT
import numpy as np
import matplotlib.pyplot as plt
import torch


def get_target(amplitude, phase, x):
    """Get the corresponding y value from an x value on a transformed sinusoidal function.

    :param amplitude: Amplitude of the wave
    :param phase: Phase shift of the wave
    :param x: X value on the sinusoid
    :return: Corresponding y value from the sinusoid
    """
    assert abs(x) <= 5 and 0.1 <= amplitude <= 5 and 0 <= phase <= np.pi
    return amplitude * np.sin(phase + x)


def get_sinusoid_data(amplitude, phase, num):
    """Randomly generate a sinusoidal dataset.

    :param amplitude: Amplitude of the wave
    :param phase: Phase shift of the wave
    :param num: Number of datapoints
    :return: Lists of x values and y values
    """
    examples = np.random.uniform(-5, 5, num)
    labels = np.array([get_target(amplitude, phase, x) for x in examples])
    return examples, labels


def scatter_plot(x, y, x_title="X Value", y_title="Y Value", title='Scatter Plot of X vs Y'):
    """Plot ordered pairs.

    :param x: List of x values
    :param y: List of y values
    :param x_title: X axis title
    :param y_title: Y axis title
    :param title: Plot title
    :return: None
    """
    plt.scatter(x, y, color='b', marker='o')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    y_ticks = [i for i in range(5)]
    plt.yticks(y_ticks)
    plt.title(title)
    plt.grid(True)
    plt.show()


def scatter_plot_two_datasets(x1, y1, x2, y2, label_1="Train", label_2="Test", x_title="X Value", y_title="Y Value", title='Scatter Plot of Two Datasets'):
    """Plot two sets of ordered pairs.

    :param x1: List of x values of dataset 1
    :param y1: List of y values of dataset 1
    :param x2: List of x values of dataset 2
    :param y2: List of x values of dataset 2
    :param label_1: Dataset 1 name
    :param label_2: Dataset 2 name
    :param x_title: X axis title
    :param y_title: Y axis title
    :param title: Plot title
    :return: None
    """
    plt.scatter(x1, y1, color='b', marker='o', label=label_1)
    plt.scatter(x2, y2, color='r', marker='s', label=label_2)

    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def get_train_and_test_data(dataset_size, training_set_ratio, amplitude, phase):
    """Randomly generate sinusoidal train and test datasets.

    :param dataset_size: Total number of datapoints
    :param training_set_ratio: Portion of
    :param amplitude: Amplitude of the wave
    :param phase: Phase shift of the wave
    :return: Lists of train examples, train labels, test examples, and test labels
    """
    training_set_size = int(dataset_size * training_set_ratio)
    examples, labels = get_sinusoid_data(amplitude, phase, dataset_size)

    train_examples = examples[:training_set_size]
    train_labels = labels[:training_set_size]
    test_examples = torch.from_numpy(examples[training_set_size:]).to(torch.float32).reshape(
        (int(np.ceil(dataset_size * (1 - training_set_ratio))), 1))
    test_labels = labels[training_set_size:]

    return train_examples, train_labels, test_examples, test_labels