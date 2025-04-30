import numpy as np
import matplotlib.pyplot as plt

# Load returns
returns_raw = np.loadtxt("returns_raw.txt")
returns_tile = np.loadtxt("returns_tile.txt")

# Rolling mean helper
def rolling_mean(data, window=25):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# --- Plot 1: Raw Features ---
plt.figure(figsize=(10, 5))
plt.plot(returns_raw, label="Raw Return", alpha=0.4)
plt.plot(rolling_mean(returns_raw), label="Raw Return (Rolling Mean)", color="green")
plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.title("MountainCar with Raw Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("eq1_a.png")
plt.show()

# --- Plot 2: Tile Features ---
plt.figure(figsize=(10, 5))
plt.plot(returns_tile, label="Raw Return", alpha=0.4)
plt.plot(rolling_mean(returns_tile), label="Tile Return (Rolling Mean)", color="blue")
plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.title("MountainCar with Tile Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("eq1_b.png")
plt.show()
