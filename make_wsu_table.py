from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, QTable, vstack
import astropy.units as u

import large_cubes
reload(large_cubes)

#gvis = u.def_unit('Gvis',namespace=globals())
#u.add_enabled_units([gvis])

result_c7_mous = QTable.read('data/wsu_datarates_mit_per_mous_cycle7_20230124.ecsv')
result_c8_mous = QTable.read('data/wsu_datarates_mit_per_mous_cycle8_20230124.ecsv')

# remove columns with problematic units
result_c7_mous.remove_column('wsu_visrate_early_stepped2_typical')
result_c7_mous.remove_column('wsu_nvis_later_2x_stepped2_typical_target_tot')
result_c7_mous.remove_column('wsu_nvis_later_2x_stepped2_typical_cal')
result_c7_mous.remove_column('wsu_nvis_later_2x_stepped2_typical_total')


result_c8_mous.remove_column('wsu_visrate_early_stepped2_typical')
result_c8_mous.remove_column('wsu_nvis_later_2x_stepped2_typical_target_tot')
result_c8_mous.remove_column('wsu_nvis_later_2x_stepped2_typical_cal')
result_c8_mous.remove_column('wsu_nvis_later_2x_stepped2_typical_total')


result_mous = vstack([result_c7_mous,result_c8_mous])

early_stats = large_cubes.calc_wsu_stats(result_mous,stage='early')
large_cubes.make_wsu_stats_table(early_stats,fileout='data/wsu_early_stats.csv')

later_2x_stats = large_cubes.calc_wsu_stats(result_mous,stage='later_2x')
large_cubes.make_wsu_stats_table(later_2x_stats,fileout='data/wsu_later_2x_stats.csv')

later_4x_stats = large_cubes.calc_wsu_stats(result_mous,stage='later_4x')
large_cubes.make_wsu_stats_table(later_4x_stats,fileout='data/wsu_later_4x_stats.csv')



