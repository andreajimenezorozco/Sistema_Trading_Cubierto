
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import pandas as pd

def load_data(start: int = 2007, end: int = 2020, freq: str = 'D'):
    column_names = ["TimeStamp", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame()
    for year in range(start, end + 1):
        file = 'files/ME_' + str(year) + '.csv'
        maindf = pd.read_csv(file,
                             header=1,
                             names=column_names,
                             parse_dates=["TimeStamp"],
                             index_col=["TimeStamp"])

        sampled_df = maindf.resample(freq).agg({'open': 'first',
                                                'close': 'last',
                                                'high': 'max',
                                                'low': 'min',
                                                'volume': 'sum'})

        sampled_df = sampled_df[sampled_df.open > 0]
        if freq == 'D':
            sampled_df.index = sampled_df.index.date
        data = data.append(sampled_df)
    return data