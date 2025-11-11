from numpy import mean, var
import os
import datetime

logs_dir = './logs/'
log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]

def mean_var_dict(d):
    prec = 5
    out_d = {}
    for k,v in d.items():
        out_d[k + '_mean'] = round(mean(v), prec)
        out_d[k + '_var'] = round(var(v), prec)
    return out_d

def compute_stats(filename):
    lines = ''
    d = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip()
        if not l.startswith("Step"):
            continue
        idx_start_dict = l.index('{')
        dct = eval(l[idx_start_dict:])
        for k,v in dct.items():
            if k not in d:
                d[k] = [v]
            else:
                d[k].append(v)
    return filename.split('/')[-1] + ': ' + str(mean_var_dict(d)) + '\n'


with open(os.path.join(logs_dir, 'summary_stats'), "w+") as g:
    g.write("Stats summary started at ")
    g.write(str(datetime.datetime.now()))
    g.write('\n')
    for f in log_files:
        g.write(compute_stats(os.path.join(logs_dir, f)))
print('Stats summary done !')
