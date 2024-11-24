# Note: Portions of this code were generated using ChatGPT
import numpy as np
import matplotlib.pyplot as plt


def get_target(amplitude, phase, x):
    assert abs(x) <= 5 and 0.1 <= amplitude <= 5 and 0 <= phase <= np.pi
    return amplitude * np.sin(phase + x)


def get_dataset(amplitude, phase, num):
    examples = np.random.uniform(-5, 5, num)
    labels = np.array([get_target(amplitude, phase, x) for x in examples])
    return examples, labels


def scatter_plot_xy(x, y):
    plt.scatter(x, y, color='b', marker='o')
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.title('Scatter Plot of X vs Y')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    x, y = get_dataset(0.1, 3.13, 500)
    scatter_plot_xy(x, y)
