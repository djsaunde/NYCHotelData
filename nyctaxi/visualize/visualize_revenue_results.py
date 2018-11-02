import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results_path = os.path.join('..', '..', 'results')


def main():
    naive = pd.read_csv(os.path.join(results_path, 'naive_revenue_results', 'results.csv'))
    fixed_effects = pd.read_csv(os.path.join(results_path, 'fixed_effects_revenue_results', 'results.csv'))
    taxi_no_fixed_effects = pd.read_csv(os.path.join(results_path, 'taxi_no_fixed_effects_revenue_results', 'results.csv'))
    taxi = pd.read_csv(os.path.join(results_path, 'taxi_revenue_results', 'results.csv'))

    # Train MSE.
    plt.figure()

    # Naive.
    naive_train_mse = naive['Train MSE'].iloc[0]
    naive_train_mse_std = naive['Train MSE Std.'].iloc[0]

    ax = sns.lineplot(range(25, 325, 25), [naive_train_mse] * 12)
    ax.errorbar(
        range(25, 325, 25), [naive_train_mse] * 12, yerr=[naive_train_mse_std] * 12, fmt='-o', label='Naive train MSE'
    )

    # Fixed effects.
    fixed_effects_train_mse = fixed_effects['Train MSE'].iloc[0]
    fixed_effects_train_mse_std = fixed_effects['Train MSE Std.'].iloc[0]

    ax = sns.lineplot(range(25, 325, 25), [fixed_effects_train_mse] * 12)
    ax.errorbar(
        range(25, 325, 25), [fixed_effects_train_mse] * 12,
        yerr=[fixed_effects_train_mse_std] * 12, fmt='-o', label='Fixed effects train MSE'
    )

    # Taxi without fixed effects.
    ax = sns.lineplot(range(25, 325, 25), taxi_no_fixed_effects['Train MSE'])
    ax.errorbar(
        range(25, 325, 25), taxi_no_fixed_effects['Train MSE'].data,
        yerr=taxi_no_fixed_effects['Train MSE Std.'].data, fmt='-o',
        label='Taxi (no fixed effects) train MSE'
    )

    # Taxi with fixed effects.
    ax = sns.lineplot(range(25, 325, 25), taxi['Train MSE'])
    ax.errorbar(
        range(25, 325, 25), taxi['Train MSE'].data, yerr=taxi['Train MSE Std.'].data,
        fmt='-o', label='Taxi (with fixed effects) train MSE'
    )

    plt.xlabel('Distance threshold in feet')
    plt.title('Train MSE regression comparison')
    plt.legend()

    plt.savefig(os.path.join('..', '..', 'plots', 'train_mse_comp.png'))

    # Train R^2.
    plt.figure()

    # Naive.
    naive_train_r2 = naive['Train R^2'].iloc[0]
    naive_train_r2_std = naive['Train R^2 Std.'].iloc[0]

    ax = sns.lineplot(range(25, 325, 25), [naive_train_r2] * 12)
    ax.errorbar(
        range(25, 325, 25), [naive_train_r2] * 12, yerr=[naive_train_r2_std] * 12, fmt='-o', label='Naive train R^2'
    )

    # Fixed effects.
    fixed_effects_train_r2 = fixed_effects['Train R^2'].iloc[0]
    fixed_effects_train_r2_std = fixed_effects['Train R^2 Std.'].iloc[0]

    ax = sns.lineplot(range(25, 325, 25), [fixed_effects_train_r2] * 12)
    ax.errorbar(
        range(25, 325, 25), [fixed_effects_train_r2] * 12, yerr=[fixed_effects_train_r2_std] * 12,
        fmt='-o', label='Fixed effects train R^2'
    )

    # Taxi without fixed effects.
    ax = sns.lineplot(range(25, 325, 25), taxi_no_fixed_effects['Train R^2'])
    ax.errorbar(
        range(25, 325, 25), taxi_no_fixed_effects['Train R^2'].data,
        yerr=taxi_no_fixed_effects['Train R^2 Std.'].data, fmt='-o',
        label='Taxi (no fixed effects) train R^2'
    )

    # Taxi with fixed effects.
    ax = sns.lineplot(range(25, 325, 25), taxi['Train R^2'])
    ax.errorbar(
        range(25, 325, 25), taxi['Train R^2'].data, yerr=taxi['Train R^2 Std.'].data,
        fmt='-o', label='Taxi (with fixed effects) train R^2'
    )

    plt.xlabel('Distance threshold in feet')
    plt.title('Train R^2 regression comparison')
    plt.legend()

    plt.savefig(os.path.join('..', '..', 'plots', 'train_r2_comp.png'))

    # Test MSE.
    plt.figure()

    # Naive.
    naive_test_mse = naive['Test MSE'].iloc[0]
    naive_test_mse_std = naive['Test MSE Std.'].iloc[0]

    ax = sns.lineplot(range(25, 325, 25), [naive_test_mse] * 12)
    ax.errorbar(
        range(25, 325, 25), [naive_test_mse] * 12, yerr=[naive_test_mse_std] * 12, fmt='-o', label='Naive test MSE'
    )

    # Fixed effects.
    fixed_effects_test_mse = fixed_effects['Test MSE'].iloc[0]
    fixed_effects_test_mse_std = fixed_effects['Test MSE Std.'].iloc[0]

    ax = sns.lineplot(range(25, 325, 25), [fixed_effects_test_mse] * 12)
    ax.errorbar(
        range(25, 325, 25), [fixed_effects_test_mse] * 12,
        yerr=[fixed_effects_test_mse_std] * 12, fmt='-o', label='Fixed effects test MSE'
    )

    # Taxi without fixed effects.
    ax = sns.lineplot(range(25, 325, 25), taxi_no_fixed_effects['Test MSE'])
    ax.errorbar(
        range(25, 325, 25), taxi_no_fixed_effects['Test MSE'].data,
        yerr=taxi_no_fixed_effects['Test MSE Std.'].data, fmt='-o',
        label='Taxi (no fixed effects) test MSE'
    )

    # Taxi with fixed effects.
    ax = sns.lineplot(range(25, 325, 25), taxi['Test MSE'])
    ax.errorbar(
        range(25, 325, 25), taxi['Test MSE'].data, yerr=taxi['Test MSE Std.'].data,
        fmt='-o', label='Taxi (with fixed effects) test MSE'
    )

    plt.xlabel('Distance threshold in feet')
    plt.title('Test MSE regression comparison')
    plt.legend()

    plt.savefig(os.path.join('..', '..', 'plots', 'test_mse_comp.png'))

    # Test R^2.
    plt.figure()

    # Naive.
    naive_test_r2 = naive['Test R^2'].iloc[0]
    naive_test_r2_std = naive['Test R^2 Std.'].iloc[0]

    ax = sns.lineplot(range(25, 325, 25), [naive_test_r2] * 12)
    ax.errorbar(
        range(25, 325, 25), [naive_test_r2] * 12, yerr=[naive_test_r2_std] * 12, fmt='-o', label='Naive test R^2'
    )

    # Fixed effects.
    fixed_effects_test_r2 = fixed_effects['Test R^2'].iloc[0]
    fixed_effects_test_r2_std = fixed_effects['Test R^2 Std.'].iloc[0]

    ax = sns.lineplot(range(25, 325, 25), [fixed_effects_test_r2] * 12)
    ax.errorbar(
        range(25, 325, 25), [fixed_effects_test_r2] * 12, yerr=[fixed_effects_test_r2_std] * 12,
        fmt='-o', label='Fixed effects test R^2'
    )

    # Taxi without fixed effects.
    ax = sns.lineplot(range(25, 325, 25), taxi_no_fixed_effects['Test R^2'])
    ax.errorbar(
        range(25, 325, 25), taxi_no_fixed_effects['Test R^2'].data,
        yerr=taxi_no_fixed_effects['Test R^2 Std.'].data, fmt='-o',
        label='Taxi (no fixed effects) test R^2'
    )

    # Taxi with fixed effects.
    ax = sns.lineplot(range(25, 325, 25), taxi['Test R^2'])
    ax.errorbar(
        range(25, 325, 25), taxi['Test R^2'].data, yerr=taxi['Test R^2 Std.'].data,
        fmt='-o', label='Taxi (with fixed effects) test R^2'
    )

    plt.xlabel('Distance threshold in feet')
    plt.title('Test R^2 regression comparison')
    plt.legend()

    plt.savefig(os.path.join('..', '..', 'plots', 'test_r2_comp.png'))

    plt.show()


if __name__ == '__main__':
    main()