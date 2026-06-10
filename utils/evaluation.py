import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_model(model, test_ds, class_names, top_k=(1, 3, 5), plot_cm=True):
    """Evaluate a Keras classifier on a held-out dataset.

    Computes Top-k accuracy for the requested values of k, a full per-class
    classification report (precision / recall / F1) and the confusion matrix,
    then lists the ten most frequent off-diagonal confusions.

    Args:
        model: Trained Keras model returning softmax probabilities.
        test_ds: ``tf.data.Dataset`` yielding ``(images, labels)`` batches
            where ``labels`` are integer class indices.
        class_names (list[str]): Ordered class labels.
        top_k (tuple[int, ...]): Values of k for Top-k accuracy. Defaults to
            ``(1, 3, 5)``.
        plot_cm (bool): If True, render the confusion-matrix figure.

    Returns:
        dict: Results dictionary with keys ``y_true``, ``y_pred``,
        ``y_pred_probs``, ``top_k`` (dict), ``macro_f1``, ``weighted_f1``,
        ``confusion_matrix`` (ndarray) and ``top_confusions``
        (list of ``(true_class, pred_class, count)`` tuples).
    """
    y_pred_probs, y_true = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred_probs.append(probs)
        y_true.append(labels.numpy())

    y_pred_probs = np.concatenate(y_pred_probs)
    y_true = np.concatenate(y_true)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Top-k accuracies
    top_k_results = {}
    for k in top_k:
        top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
        hits = np.any(top_k_preds == y_true[:, None], axis=1)
        top_k_results[k] = float(np.mean(hits))

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("--------- Classification report ---------")
    topk_str = "  ".join(f"Top-{k}: {v:.4f}" for k, v in top_k_results.items())
    print(topk_str)
    print(f"macro-F1: {macro_f1:.4f}    weighted-F1: {weighted_f1:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    if plot_cm:
        print("--------- Confusion matrix ---------")
        fig, ax = plt.subplots(figsize=(16, 14))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, xticks_rotation=90, values_format="d", cmap="Blues")
        plt.tight_layout()
        plt.show()

    # Top-10 off-diagonal confusions (systematic errors worth discussing)
    cm_off = cm.astype(float).copy()
    np.fill_diagonal(cm_off, 0)

    print("--------- Top 10 misclassifications ---------")
    print(f"{'True class':>30s}  as  {'Predicted class':<30s}  Count")
    print("-" * 90)
    top_confusions = []
    for _ in range(10):
        i, j = np.unravel_index(cm_off.argmax(), cm_off.shape)
        count = int(cm_off[i, j])
        if count == 0:
            break
        print(f"{class_names[i]:>30s}  as  {class_names[j]:<30s}  {count}")
        top_confusions.append((class_names[i], class_names[j], count))
        cm_off[i, j] = 0

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs,
        "top_k": top_k_results,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "top_confusions": top_confusions,
    }
