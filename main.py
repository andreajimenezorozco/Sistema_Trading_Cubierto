
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import data as dt
import functions as fn

#Leer los archivos
data_t = dt.load_data(start=2010, end=2020, freq='D')

#crear variables artificiales - indicadores financieros y estadísticos
data_t = fn.add_all_features(data_t)

#crear variables artificiales - transformaciones matemáticas y change point
data_t = fn.create(data_t)

#selección de variables
data_t = fn.ANOVA_importance(data_t,0.79,'Label')

#modelado - ANN