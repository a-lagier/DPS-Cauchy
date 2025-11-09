import os
from numpy import mean
import datetime

def print_dict(dct, t=''):
    s = ''
    for k,v in dct.items():
        if isinstance(v, dict):
            s += t + k + ': \n' + print_dict(v, t=t+'  ')
        else:
            s += t + k + ': ' + str(v) + '\n'
    return s

class Logger():

    def __init__(self, keys: list, log_name='logging.log', prec: int = 5):
        self.log_dir = './logs/'
        self.log_file = os.path.join(self.log_dir, log_name)
        self.logs = {k:[] for k in keys}
        self.prec = prec
        self.len_logs = 0

        self.reset_log()
    
    def update_stats(self, **kwargs):
        for metric in kwargs:
            self.logs[metric].append(kwargs[metric])
        self.len_logs += 1
    
    def reset_log(self):
        open(self.log_file, 'w').close()

    def write_config(self, cfg):
        with open(self.log_file, 'a') as f:
            f.write("Experiment started at {}\n".format(datetime.datetime.now()))
            f.write(print_dict(cfg))

    def write_step(self, step):
        step_stats = {k: round(v[-1], self.prec) for (k,v) in self.logs.items()}

        log = 'Step {}: {}'.format(step, step_stats)
        
        with open(self.log_file, 'a') as f:
            f.write(log + '\n')
        print(log)
    
    def write_end_step(self):
        step_stats = {k: round(mean(v), self.prec) for (k,v) in self.logs.items()}

        log = 'Experiment results: {}'.format(step_stats)
        
        with open(self.log_file, 'a') as f:
            f.write(log + '\n')
        print(log)