import numpy as np
import matplotlib.pyplot as plt
import pso

if __name__ == "__main__":
    s = pso.Swarm(10, 2, -1, 1, 0, 0, eval_func=lambda s: s[0]**2 + s[1]**2)

    scores = []

    for _ in range(100):
        scores.append(s.get_gbest_score())
        s.update()

    plt.plot(scores)
    plt.show()
