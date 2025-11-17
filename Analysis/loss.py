"""
Script for plotting loss of the checkpoints of a model.
"""

import pandas as pd
import matplotlib.pyplot as plt


def gather_data(path: str):
    """
    Loads a CSV file into a DataFrame, assigns column names for training 
    metadata (Epoch, Step, Train_Loss, Test_Loss), and returns it.
    """
    df = pd.read_csv(path, delimiter=",", header=None)
    df.columns = ["Epoch", "Step", "Train_Loss", "Test_Loss"]
    return df


def plot_data(df: pd.DataFrame):
    """
    Parses training and test loss values from a dataframe, aligns them by step, 
    and plots average epoch loss over steps using matplotlib.
    """
    train_loss = [float(entry.split(" ")[-1]) for entry in df["Train_Loss"].tolist()]
    test_loss = [float(entry.split(" ")[-1]) for entry in df["Test_Loss"].tolist()]
    step = [entry.split(" ")[-1] for entry in df["Step"].tolist()]

    train_loss[0] = test_loss[0]

    plt.plot(step, train_loss)
    plt.xlabel("Step")
    plt.ylabel("Avg Epoch Loss")
    plt.xticks(rotation=-45)
    plt.show()


if __name__ == "__main__":
    PATH = "./log.txt"
    df = gather_data(PATH)
    plot_data(df)
