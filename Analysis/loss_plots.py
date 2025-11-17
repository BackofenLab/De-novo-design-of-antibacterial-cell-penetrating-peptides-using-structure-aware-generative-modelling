import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(filename):
    epochs = []
    train_losses = []
    test_losses = []
    
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(r"Epoch: (\d+), Step: \d+, Train Loss: ([\d\.nan\-]+), Test Loss: ([\d\.nan\-]+)", line)
            if match:
                epoch = int(match.group(1))
                train_loss = match.group(2)
                test_loss = match.group(3)

                # Convert to float (handle nan)
                train_loss = float(train_loss) if train_loss != "nan" else np.nan
                test_loss = float(test_loss) if test_loss != "nan" else np.nan

                epochs.append(epoch)
                train_losses.append(train_loss)
                test_losses.append(test_loss)

    return epochs, train_losses, test_losses

def plot_losses(epochs, train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PeptideEmbedNet (non-nat.)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Change 'log.txt' to your actual filename
    log_file = "../Checkpoints/log.txt"
    epochs, train_losses, test_losses = parse_log_file(log_file)
    plot_losses(epochs, train_losses, test_losses)

