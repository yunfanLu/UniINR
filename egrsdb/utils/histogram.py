import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(data, path, bins=150, title="Histogram", xlabel="Value", ylabel="Frequency"):
    plt.hist(data, bins=bins, range=(0, 1), edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)


if __name__ == "__main__":
    data = [0.1, 0.2, 0.35, 0.5, 0.6, 0.85, 0.9, 0.95]

    plot_histogram(data, "test_histogram.png")
