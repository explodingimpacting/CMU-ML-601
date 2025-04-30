import numpy as np
import matplotlib.pyplot as plt

# Load returns
returns_tile_replay = np.loadtxt("returns_tile_replay.txt")

# Rolling mean helper
def rolling_mean(data, window=25):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# Plot
plt.figure(figsize=(10, 5))
plt.plot(returns_tile_replay, label="Sum Raw Return", alpha=0.4)
plt.plot(rolling_mean(returns_tile_replay), label="Rolling Mean", color="purple")
plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.title("MountainCar with Tile Features & Replay Buffer")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("eq2.png")
plt.show()
