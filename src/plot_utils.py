import os
import matplotlib.pyplot as plt


def plot_training_history(history, output_dir="output/plots"):
    """
    Plot training accuracy/loss and validation accuracy/loss.
    """

    os.makedirs(output_dir, exist_ok=True)

    accuracy = history.history["accuracy"]
    validation_accuracy = history.history["val_accuracy"]

    loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    epochs = range(1, len(accuracy) + 1)

    # Accuracy graph
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, validation_accuracy, label="Validation Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    accuracy_plot_path = os.path.join(output_dir, "accuracy_plot.png")
    plt.savefig(accuracy_plot_path)
    plt.close()

    # Loss graph
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, validation_loss, label="Validation Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    loss_plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()

    return accuracy_plot_path, loss_plot_path