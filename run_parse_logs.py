from importlib import reload

import parse_logs
reload(parse_logs)

test = parse_logs.parse_pipe_casalog('/Users/akepley/Dropbox/Support/naasc/WSU/big_cubes/data/felix_logfiles/Cycle7/logfiledirectory/member.uid___A001_X3360_X28.hifa_calimage.weblog.tgz/casa-20221005-173548286714693.log')

cycle7_dir = '/Users/akepley/Dropbox/Support/naasc/WSU/big_cubes/data/felix_logfiles/Cycle7/logfiledirectory'

result = parse_logs.parse_all_pipe_casalogs(cycle7_dir)

