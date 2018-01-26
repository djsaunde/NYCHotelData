import os
import datetime
import numpy as np
import pandas as pd

print '... Loading hotel identification Excel worksheet.'
hotel_id_mapping = pd.read_excel(os.path.join('..', 'data', 'ID_to_hotel.xlsx'))

print '... Loading daily hotel capacity data.'
daily_capacity_data = pd.read_excel(os.path.join('..', 'data', 'SHARE_Property_Daily_Data_Dan.xlsx'))

print '... Converting hotel identification worksheet to a dictionary object.'
hotel_id_mapping = { int(id_) : hotel_name for (id_, hotel_name) in zip(hotel_id_mapping['Share ID'], hotel_id_mapping['Name']) if not np.isnan(id_) }

print '... Mapping daily capacity data from anonymous ID to hotel name.'
daily_capacity_data['Share ID'] = pd.Series([ hotel_id_mapping[int(datum)] for datum in daily_capacity_data['Share ID'] ])

print '... Converting string dates to Python datetime objects.'
daily_capacity_data['Date'] = pd.Series([ datetime.datetime(int(str(datum)[0:4]), int(str(datum)[4:6]), int(str(datum)[6:8])) for datum in daily_capacity_data['Date'] ])

daily_capacity_data.to_csv(os.path.join('..', 'data', 'Unmasked Daily Capacity.csv'))
