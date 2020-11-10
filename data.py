
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

def f_leer_archivo(param_archivo: str):
    # print(param_archivo[-4:])
    if param_archivo[-3:] == 'csv':
        data = pd.read_csv(param_archivo, low_memory=True)
        return data
    elif param_archivo[-4:] == 'xlsx' or param_archivo[-3:] == 'xls':
        data = pd.read_excel(param_archivo)
        return data
    else:
        data = "Ingresa un formato Excel válido"
        return data