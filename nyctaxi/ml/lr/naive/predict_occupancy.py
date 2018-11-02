from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd

from datetime import date

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from nyctaxi.ml.utils import load_occupancy_data, encode_data


parser = argparse.ArgumentParser()
parser.add_argument('--start_date', type=int, nargs=3, default=[2014, 1, 1])
parser.add_argument('--end_date', type=int, nargs=3, default=[2016, 6, 30])
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--targets', type=str, nargs='+', default=['Occ'])
args = parser.parse_args()

start_date = args.start_date
end_date = args.end_date
trials = args.trials
targets = args.targets

fname = '_'.join(map(str, [start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2]]))

start_date, end_date = date(*start_date), date(*end_date)

top = os.path.join('..', '..', '..', '..')
data_path = os.path.join(top, 'data')
predictions_path = os.path.join(data_path, 'naive_logreg_predictions', fname)
results_path = os.path.join(top, 'results', 'naive_logreg_results')

for path in [predictions_path, results_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

# Load daily capacity data.
df = load_occupancy_data(data_path, start_date, end_date)
observations, targets = encode_data(df, obs=('IDs', ), targets=targets)

# Save (observations, targets) to disk.
np.save(os.path.join(data_path, 'naive_observations.npy'), observations)
np.save(os.path.join(data_path, 'targets.npy'), targets)

train_scores = []
test_scores = []
train_mses = []
test_mses = []

for i in range(trials):  # Run 5 independent realizations of training / test.
    print('\nTraining, testing model %d / %d' % (i + 1, trials))

    # Randomly permute the data to remove sequence biasing.
    p = np.random.permutation(targets.shape[0])
    x = observations[p]
    y = targets[p]

    # Split the data into (training, test) subsets.
    split = int(0.8 * len(y))
    train_x, train_y = x[:split], y[:split]
    test_x, test_y = x[split:], y[split:]

    print('Creating and training linear regression model.')

    model = LinearRegression().fit(train_x, train_y)

    print('Training complete. Getting predictions and calculating R^2, MSE.')

    train_scores.append(model.score(train_x, train_y))
    test_scores.append(model.score(test_x, test_y))

    train_y_hat = model.predict(train_x)
    test_y_hat = model.predict(test_x)

    train_mses.append(mean_squared_error(train_y, train_y_hat))
    test_mses.append(mean_squared_error(test_y, test_y_hat))

    print()
    print('*** Results on %d / %d trial ***' % (i + 1, trials))
    print()
    print('Training MSE: %.8f' % train_mses[-1])
    print('Training R^2: %.8f' % train_scores[-1])
    print()
    print('Test MSE: %.8f' % test_mses[-1])
    print('Test R^2: %.8f' % test_scores[-1])
    print()

    np.save(os.path.join(predictions_path, 'train_targets_%d.npy' % i), train_y)
    np.save(os.path.join(predictions_path, 'train_predictions_%d.npy' % i), train_y_hat)

    np.save(os.path.join(predictions_path, 'test_targets_%d.npy' % i), test_y)
    np.save(os.path.join(predictions_path, 'test_predictions_%d.npy' % i), test_y_hat)

print()
print('Mean, standard deviation of training MSE: %.8f $\pm$ %.8f' % (np.mean(train_mses), np.std(train_mses)))
print('Mean, standard deviation of training R^2: %.8f' % np.mean(train_scores))
print()
print('Mean, standard deviation of test MSE: %.8f $\pm$ %.8f' % (np.mean(test_mses), np.std(test_mses)))
print('Mean, standard deviation of test R^2: %.8f' % np.mean(test_scores))
print()
print('%.8f $\pm$ %.8f & %.8f & %.8f $\pm$ %.8f & %.8f' % (
    np.mean(train_mses), np.std(train_mses), np.mean(train_scores),
    np.mean(test_mses), np.std(test_mses), np.mean(test_scores)
))
print()

columns = ['Train MSE', 'Train MSE Std.', 'Train R^2', 'Train R^2 Std.',
           'Test MSE', 'Test MSE Std.', 'Test R^2', 'Test R^2 Std.']
data = [[np.mean(train_mses), np.std(train_mses), np.mean(train_scores), np.std(train_scores),
         np.mean(test_mses), np.std(test_mses), np.mean(test_scores), np.std(test_scores)]]

path = os.path.join(results_path, 'results.csv')
if not os.path.isfile(path):
    df = pd.DataFrame(data=data, index=[fname], columns=columns)
else:
    df = pd.read_csv(path, index_col=0)
    if not fname in df.index:
        df.append(pd.DataFrame(data=data, index=[fname], columns=columns))
    else:
        df.loc[fname] = data[0]

df.to_csv(path, index=True)
