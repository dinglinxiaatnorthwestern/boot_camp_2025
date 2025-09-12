import numpy as np

# Set parameters for the normal distribution
mean = 0
std_dev = 1
num_samples = 1000

# Generate samples
samples = np.random.normal(mean, std_dev, num_samples)

# Print first 10 samples
print(samples[:10])

normalized_samples = (samples - np.mean(samples)) / np.std(samples)
print(normalized_samples[:10])

print('Mean of normalized samples:', np.mean(normalized_samples))
print('Standard deviation of normalized samples:', np.std(normalized_samples))

# using the generated samples to simulate a simple random walk
steps = np.where(samples > 0, 1, -1)
position = np.cumsum(steps)
print('Final position after random walk:', position[-1])

# plot the random walk
import matplotlib.pyplot as plt
plt.plot(position)
plt.title('Random Walk Simulation')
plt.xlabel('Step')
plt.ylabel('Position')
plt.show()

# some more analysis


import scipy.stats as stats
k2, p = stats.normaltest(samples)
alpha = 0.05
print("p = {:g}".format(p))
if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")
import seaborn as sns
sns.histplot(samples, kde=True)
plt.show()

