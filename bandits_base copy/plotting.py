import matplotlib.pyplot as plt
import numpy as np

#load results here
lines = np.zeros(1000)
data = open("RL_EXP_OUT.dat")
data2 = open("RL_EXP_OUT1.dat")
lines = data.readlines()
lines1 = data2.readlines()
print(lines)
x = np.arange(1, 1000 + 1)
plt.plot(x, lines)
plt.plot(x, lines1)
plt.show()
