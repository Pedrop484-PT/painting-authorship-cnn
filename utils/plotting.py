import matplotlib.pyplot as plt


def plot_history(history, title, accuracy_key="accuracy", val_accuracy_key="val_accuracy"):
    """Plot training and validation loss/accuracy curves across epochs.

    Args:
        history: Keras History object returned by ``model.fit``.
        title (str): Title prefix shown on both subplots.
        accuracy_key (str): Key for the training accuracy metric stored in
            ``history.history``. Defaults to ``"accuracy"``. Use
            ``"top1_accuracy"`` when training with one-hot labels and a custom
            Top-1 metric.
        val_accuracy_key (str): Corresponding validation-accuracy key.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        ax1.plot(history.history["val_loss"], label="val")
    ax1.set_title(f"{title} (Loss)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    if accuracy_key in history.history:
        ax2.plot(history.history[accuracy_key], label="train")
    if val_accuracy_key in history.history:
        ax2.plot(history.history[val_accuracy_key], label="val")
    ax2.set_title(f"{title} (Accuracy)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()
    return fig
