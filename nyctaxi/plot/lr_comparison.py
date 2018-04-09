import os
import pandas as pd
import matplotlib.pyplot as plt

naive_path = os.path.join('..', '..', 'results', 'naive_lr_results', 'results.csv')
taxi_path = os.path.join('..', '..', 'results', 'taxi_lr_results', 'results.csv')
plots_path = os.path.join('..', '..', 'plots')

naive_df = pd.read_csv(naive_path)
taxi_df = pd.read_csv(taxi_path)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

axes[0].plot(list(naive_df['Test MSE']) * len(taxi_df['Test MSE']), 'k--', label='Naive')
axes[0].plot(taxi_df['Test MSE'], label='Taxi')

axes[1].plot(list(naive_df['Test R^2']) * len(taxi_df['Test MSE']), 'k--', label='Naive')
axes[1].plot(taxi_df['Test R^2'], label='Taxi')

for ax in axes:
	ax.legend()
	ax.grid()

axes[1].set_xticks(range(12))
axes[1].set_xticklabels(range(25, 325, 25))
axes[1].set_xlabel('Taxi trip distance threshold')

axes[0].set_ylabel('Averaged Test MSE')
axes[1].set_ylabel('Averaged Test R^2')

fig.suptitle('Test MSE and R^2: naive vs. taxi data')

plt.savefig(os.path.join(plots_path, 'lr_comparison.png'))