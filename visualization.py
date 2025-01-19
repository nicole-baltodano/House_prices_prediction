### visualization.py
import matplotlib.pyplot as plt

def plot_results(y_test, predictions, rmsle):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6, color="blue", edgecolor="k", label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             color="red", linestyle="--", linewidth=2, label="Perfect Predictions")
    plt.xlim(0, max(y_test.max(), predictions.max()))
    plt.ylim(0, max(y_test.max(), predictions.max()))
    plt.title("Actual vs Predicted Values with RMSLE", fontsize=14)
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.text(0.05 * max(y_test.max(), predictions.max()),
             0.9 * max(y_test.max(), predictions.max()),
             f"RMSLE = {rmsle:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True)
    plt.show()
