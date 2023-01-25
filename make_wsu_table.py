from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import QTable, vstack

import large_cubes
reload(large_cubes)


result_c7_mous = QTable.read('data/wsu_datarates_mit_per_mous_cycle7_20230124.ecsv')
result_c8_mous = QTable.read('data/wsu_datarates_mit_per_mous_cycle8_20230124.ecsv')

result_mous = vstack([result_c7_mous, result_c8_mous])



early_stats = large_cubes.calc_wsu_stats(result_c7_mous,stage='early')




large_cubes.make_wsu_stats_table(early_stats,fileout='data/wsu_early_stats.csv')

