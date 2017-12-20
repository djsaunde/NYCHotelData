import os
import csv
import argparse

from util import daterange
from datetime import timedelta, date, datetime

output_path = os.path.join('..', 'data', 'daily_distributions')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--start_date', type=int, nargs=3, default=[2013, 1, 1], help='The day on \
												which to start looking for satisfying coordinates.')
	parser.add_argument('--end_date', type=int, nargs=3, default=[2013, 1, 5], help='The day on \
												which to stop looking for satisfying coordinates.')
	parser.add_argument('--coord_type', type=str, default='pickups', help='The type of coordinates \
											to look for (one of "pickups", "dropoffs", or "both").')
	parser.add_argument('--distance', type=int, default=100, help='The distance (in feet) from hotels \
													for which to look for satisfying taxicab trips.')

	args = vars(parser.parse_args())
	locals().update(args)

	if coord_type == 'pickups':
		coord_types = ['pickups']
	elif coord_type == 'dropoffs':
		coord_types = ['dropoffs']
	elif coord_type == 'both':
		coord_types = ['pickups', 'dropoffs']

	start_date, end_date = date(*start_date), date(*end_date)

	fname = '_'.join([ '_'.join(coord_types), str(distance), str(start_date), str(end_date) ]) + '.csv'
	with open(os.path.join(output_path, fname), 'w') as to_write:
		writer = csv.writer(to_write)

		for day_idx, date in enumerate(daterange(start_date, end_date)):
			print '\n*** Date:', date, '***\n'

			current_fname = '_'.join([ '_'.join(coord_types), str(distance), str(date) ]) + '.csv'
			with open(os.path.join(output_path, current_fname), 'r') as to_read:
				reader = csv.reader(to_read)
				for row in reader:
					writer.writerow([date] + row)