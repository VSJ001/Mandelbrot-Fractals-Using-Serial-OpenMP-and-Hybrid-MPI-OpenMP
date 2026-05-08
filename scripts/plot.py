import matplotlib.pyplot as plt
import numpy as np

# Provided Data
resolutions = ['1024x768', '1920x1080', '3840x2160']
iterations = ['500', '1000', '5000']

serial_times = np.array([[0.726, 1.417, 6.928],
                         [1.912, 3.731, 18.259],
                         [7.638, 14.918, 72.993]])

openmp_times = np.array([[0.182, 0.355, 1.734],
                         [0.479, 0.934, 4.570],
                         [1.913, 3.734, 18.272]])

hybrid_times = np.array([[0.150, 0.301, 1.447],
                         [0.394, 0.794, 3.809],
                         [1.571, 3.085, 15.212]])

# Speedup and Efficiency calculations
openmp_speedup = serial_times / openmp_times
hybrid_speedup = serial_times / hybrid_times

openmp_efficiency = openmp_speedup / 4  # 4 threads
hybrid_efficiency = hybrid_speedup / 8  # 4 ranks × 2 threads

# Plotting graphs
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Execution Time
for idx, res in enumerate(resolutions):
    axes[0].plot(iterations, serial_times[idx], '-o', label=f'Serial {res}')
    axes[0].plot(iterations, openmp_times[idx], '--o', label=f'OpenMP {res}')
    axes[0].plot(iterations, hybrid_times[idx], ':o', label=f'Hybrid {res}')
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('Execution Time (s)')
axes[0].set_title('Execution Time Comparison')
axes[0].legend()
axes[0].grid()

# Speedup
for idx, res in enumerate(resolutions):
    axes[1].plot(iterations, openmp_speedup[idx], '-o', label=f'OpenMP {res}')
    axes[1].plot(iterations, hybrid_speedup[idx], '--o', label=f'Hybrid {res}')
axes[1].set_xlabel('Iterations')
axes[1].set_ylabel('Speedup')
axes[1].set_title('Speedup Comparison')
axes[1].legend()
axes[1].grid()

# Efficiency
for idx, res in enumerate(resolutions):
    axes[2].plot(iterations, openmp_efficiency[idx]*100, '-o', label=f'OpenMP {res}')
    axes[2].plot(iterations, hybrid_efficiency[idx]*100, '--o', label=f'Hybrid {res}')
axes[2].set_xlabel('Iterations')
axes[2].set_ylabel('Efficiency (%)')
axes[2].set_title('Efficiency Comparison')
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.show()
