from __future__ import print_function

import os
import datetime
import numpy as np
import pandas as pd

print('... Loading hotel identification Excel worksheet.')
hotel_id_mapping = pd.read_excel(os.path.join('..', 'data', 'Final hotel Identification.xlsx'))

print('... Loading daily hotel capacity and price data.')
capacity_and_price_data = pd.read_csv(os.path.join('..', 'data', 'occ and price data.csv'))

print('... Converting hotel identification worksheet to a dictionary object.')
hotel_id_mapping = { int(id_) : hotel_name for (id_, hotel_name) in zip(hotel_id_mapping['Share ID'], hotel_id_mapping['Name']) if not np.isnan(id_) }

print('... Mapping daily capacity and price data from anonymous ID to hotel name.')
capacity_and_price_data['Share ID'] = pd.Series([ hotel_id_mapping[int(datum)] for datum in capacity_and_price_data['Share ID'] ])

print('... Converting string dates to Python datetime objects.')
capacity_and_price_data['Date'] = pd.Series([ datetime.datetime(int(str(datum)[0:4]), int(str(datum)[4:6]), int(str(datum)[6:8])) for datum in capacity_and_price_data['Date'] ])

capacity_and_price_data.to_csv(os.path.join('..', 'data', 'Unmasked Capacity and Price Data.csv'))
