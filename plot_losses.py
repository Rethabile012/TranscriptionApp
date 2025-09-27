import pandas as pd
import matplotlib.pyplot as plt

# Load training history
history = pd.read_csv("training_history.csv")

# Plot losses
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig("loss_curve.png")
print("Saved plot as loss_curve.png")
