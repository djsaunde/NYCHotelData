import os
import argparse
import numpy as np
import pandas as pd

from util import *


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--distances', nargs='+', default=[25, 50, 75, 100])

	args = parser.parse_args()
	args = vars(args)

	locals().update(args)

	# Load daily capacity data
	daily_capacity_data = pd.read_csv(os.path.join('..', 'data', 'Unmasked Daily Capacity.xlsx'))