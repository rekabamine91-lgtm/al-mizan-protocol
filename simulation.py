import numpy as np
import matplotlib.pyplot as plt
from mizan_engine import MizanValue, AlMizanMLP, adaptive_zakat_rate

def run_simulation():
    # XOR dataset
    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,1,1,0]

    # Standard training (justice disabled)
    model_std = AlMizanMLP(2, [4,1])
    for p in model_std.parameters():
        p.tyranny_threshold = 1e9
        p.zakat_rate = 0.0

    losses_std = []
    for epoch in range(500):
        preds = []
        for x_vals in X:
            x = [MizanValue(v) for v in x_vals]
            pred = model_std.forward(x)
            preds.append(pred)
        loss = sum((pred - MizanValue(yi))**2 for pred, yi in zip(preds, Y))
        losses_std.append(loss.data)
        model_std.backward(loss)
        for p in model_std.parameters():
            p.data -= 0.05 * p.grad
            p.grad = 0.0

    # Constitutional training
    model_const = AlMizanMLP(2, [4,1], zakat_rate=0.025)
    losses_const = []
    tyranny_total = []
    for epoch in range(500):
        preds = []
        for x_vals in X:
            x = [MizanValue(v) for v in x_vals]
            pred = model_const.forward(x)
            preds.append(pred)
        loss = sum((pred - MizanValue(yi))**2 for pred, yi in zip(preds, Y))
        losses_const.append(loss.data)
        model_const.backward(loss)
        total_tyranny = sum(p.tyranny_count for p in model_const.parameters())
        tyranny_total.append(total_tyranny)
        for p in model_const.parameters():
            p.data -= 0.05 * p.grad
            p.grad = 0.0

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(losses_std, label="Standard (PyTorch-like)", color='red')
    plt.plot(losses_const, label="Al-Mizan Constitutional", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Comparison")

    plt.subplot(1,2,2)
    plt.plot(tyranny_total, color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Total Tyranny Events")
    plt.title("Algorithmic Tyranny under Al-Mizan")
    plt.tight_layout()
    plt.savefig("almizan_simulation.png")
    plt.show()
    print("Simulation complete. Check almizan_simulation.png")

if __name__ == "__main__":
    run_simulation()
