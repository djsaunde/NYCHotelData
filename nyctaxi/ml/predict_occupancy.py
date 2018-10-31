from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd

from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from nyctaxi.ml.utils import load_merged_data, load_occupancy_data, encode_data, LogitRegression

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='naive')
parser.add_argument('--start_date', type=int, nargs=3, default=[2014, 1, 1])
parser.add_argument('--end_date', type=int, nargs=3, default=[2016, 6, 30])
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--distance', type=int, default=100)
parser.add_argument('--trip_type', type=str, default='pickups')

args = parser.parse_args()

dataset = args.dataset
start_date = args.start_date
end_date = args.end_date
trials = args.trials
distance = args.distance
trip_type = args.trip_type

if 'taxi' in dataset:
    fname = '_'.join(map(str, [
        distance, start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2]
    ]))
else:
    fname = '_'.join(map(str, [
        start_date[0], start_date[1], start_date[2], end_date[0], end_date[1], end_date[2]
    ]))

start_date, end_date = date(*start_date), date(*end_date)

top = os.path.join('..', '..')
data_path = os.path.join(top, 'data')
preproc_data_path = os.path.join(data_path, f'all_preprocessed_{distance}')
predictions_path = os.path.join(data_path, f'{dataset}_revenue_predictions', fname)
taxi_occupancy_path = os.path.join(data_path, 'taxi_occupancy', fname)
results_path = os.path.join(top, 'results', f'{dataset}_revenue_results')
lr_results_path = os.path.join(top, 'results', f'{dataset}_lr_results')
logit_results_path = os.path.join(top, 'results', f'{dataset}_logit_results')

for path in [predictions_path, results_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

# Load daily capacity data.
if dataset == 'naive':
    df = load_occupancy_data(data_path, start_date, end_date)
    observations, targets = encode_data(df, obs=('IDs',), targets=('Occ', 'ADR', 'Revenue', 'Capacity'))
elif dataset == 'taxi_no_fixed_effects':
    df = load_merged_data(
        data_path, taxi_occupancy_path, preproc_data_path, start_date, end_date, trip_type
    )
    observations, targets = encode_data(df, obs=('IDs',), targets=('Occ', 'ADR', 'Revenue', 'Capacity'))
elif dataset == 'fixed_effects':
    df = load_occupancy_data(data_path, start_date, end_date)
    observations, targets = encode_data(df, obs=('IDs', 'Weekdays', 'Months', 'Years'),
                                        targets=('Occ', 'ADR', 'Revenue', 'Capacity'))
elif dataset == 'taxi':
    df = load_merged_data(
        data_path, taxi_occupancy_path, preproc_data_path, start_date, end_date, trip_type
    )
    observations, targets = encode_data(df, obs=('IDs', 'Weekdays', 'Months', 'Years'),
                                        targets=('Occ', 'ADR', 'Revenue', 'Capacity'))
else:
    raise NotImplementedError

# Save (observations, targets) to disk.
np.save(os.path.join(data_path, '{dataset}_observations.npy'), observations)
np.save(os.path.join(data_path, 'targets.npy'), targets)

train_scores = []
test_scores = []
train_mses = []
test_mses = []

# train_lr_scores = []
# test_lr_scores = []
# train_lr_mses = []
# test_lr_mses = []
#
# train_logit_scores = []
# test_logit_scores = []
# train_logit_mses = []
# test_logit_mses = []

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

    print('Creating and training occupancy regression model.')

    occ_model = LogitRegression().fit(train_x, train_y[:, 0].reshape(-1, 1))

    print('Creating and training ADR regression model.')

    adr_model = LinearRegression().fit(train_x, train_y[:, 1].reshape(-1, 1))

    print('Training complete. Getting predictions and calculating R^2, MSE.')

    train_preds = occ_model.predict(train_x) * adr_model.predict(train_x) * train_y[:, 3].reshape(-1, 1)
    test_preds = occ_model.predict(test_x) * adr_model.predict(test_x) * test_y[:, 3].reshape(-1, 1)

    train_r2 = r2_score(train_y[:, 2].reshape(-1, 1), train_preds)
    test_r2 = r2_score(test_y[:, 2].reshape(-1, 1), test_preds)

    train_scores.append(train_r2)
    test_scores.append(test_r2)

    train_mses.append(mean_squared_error(train_y[:, 2], train_preds))
    test_mses.append(mean_squared_error(test_y[:, 2], test_preds))

    print()
    print('*** Results on %d / %d trial ***' % (i + 1, trials))
    print()
    print('Training MSE: %.8f' % train_mses[-1])
    print('Training R^2: %.8f' % train_scores[-1])
    print()
    print('Test MSE: %.8f' % test_mses[-1])
    print('Test R^2: %.8f' % test_scores[-1])
    print()

    # train_lr_scores.append(r2_score(train_y[:, 1].reshape(-1, 1), adr_model.predict(train_x[:, 1].reshape(-1, 1))))
    # test_lr_scores.append(r2_score(test_y[:, 1].reshape(-1, 1), adr_model.predict(test_x[:, 1].reshape(-1, 1))))
    #
    # train_lr_mses.append(
    #     mean_squared_error(train_y[:, 1].reshape(-1, 1), adr_model.predict(train_x[:, 1].reshape(-1, 1)))
    # )
    # train_lr_mses.append(
    #     mean_squared_error(test_y[:, 1].reshape(-1, 1), adr_model.predict(test_x[:, 1].reshape(-1, 1)))
    # )
    #
    # train_lr_scores.append(r2_score(train_y[:, 0].reshape(-1, 1), adr_model.predict(train_x[:, 0].reshape(-1, 1))))
    # test_lr_scores.append(r2_score(test_y[:, 0].reshape(-1, 1), adr_model.predict(test_x[:, 0].reshape(-1, 1))))
    #
    # train_lr_mses.append(
    #     mean_squared_error(train_y[:, 0].reshape(-1, 1), adr_model.predict(train_x[:, 0].reshape(-1, 1)))
    # )
    # train_lr_mses.append(
    #     mean_squared_error(test_y[:, 0].reshape(-1, 1), adr_model.predict(test_x[:, 0].reshape(-1, 1)))
    # )

    np.save(os.path.join(predictions_path, 'train_targets_%d.npy' % i), train_y)
    np.save(os.path.join(predictions_path, 'train_predictions_%d.npy' % i), train_preds)

    np.save(os.path.join(predictions_path, 'test_targets_%d.npy' % i), test_y)
    np.save(os.path.join(predictions_path, 'test_predictions_%d.npy' % i), test_preds)

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

columns = [
    'Train MSE', 'Train MSE Std.', 'Train R^2', 'Train R^2 Std.',
    'Test MSE', 'Test MSE Std.', 'Test R^2', 'Test R^2 Std.'
]
data = [[
    np.mean(train_mses), np.std(train_mses), np.mean(train_scores), np.std(train_scores),
    np.mean(test_mses), np.std(test_mses), np.mean(test_scores), np.std(test_scores)
]]

path = os.path.join(results_path, 'results.csv')
if not os.path.isfile(path):
    df = pd.DataFrame(data=data, index=[fname], columns=columns)
else:
    df = pd.read_csv(path, index_col=0)

    if fname not in df.index:
        df = df.append(pd.DataFrame(data=data, index=[fname], columns=columns))
    else:
        df.loc[fname] = data[0]

df.to_csv(path, index=True)
