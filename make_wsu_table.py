from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, QTable, vstack
import astropy.units as u

import wsu_db
reload(wsu_db)

gvis = u.def_unit('Gvis',namespace=globals())
u.add_enabled_units([gvis])

#result_c7_mous = QTable.read('data/wsu_datarates_mit_per_mous_cycle7_20230321.ecsv')
#result_c8_mous = QTable.read('data/wsu_datarates_mit_per_mous_cycle8_20230321.ecsv')

result_c7_mous = QTable.read('data/wsu_datarates_mit_per_mous_cycle7_20230420.ecsv')
result_c8_mous = QTable.read('data/wsu_datarates_mit_per_mous_cycle8_20230420.ecsv')

result_mous = vstack([result_c7_mous,result_c8_mous])

early_stats = wsu_db.calc_wsu_stats(result_mous,stage='early')
wsu_db.make_wsu_stats_table(early_stats,fileout='data/wsu_early_stats_20230420.csv')

later_2x_stats = wsu_db.calc_wsu_stats(result_mous,stage='later_2x')
wsu_db.make_wsu_stats_table(later_2x_stats,fileout='data/wsu_later_2x_stats_20230420.csv')

later_4x_stats = wsu_db.calc_wsu_stats(result_mous,stage='later_4x')
wsu_db.make_wsu_stats_table(later_4x_stats,fileout='data/wsu_later_4x_stats_20230420.csv')



