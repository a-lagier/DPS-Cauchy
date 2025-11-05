import logging

class Logger():

    def __init__(self, keys, prec=5):
        self.logs = {k:[] for k in keys}
        self.prec = prec
        self.len_logs = 0
    
    def update_stats(self, **kwargs):
        for metric in kwargs:
            self.logs[metric].append(kwargs[metric])
        self.len_logs += 1
    
    def write_step(self, step):
        step_stats = {k: round(v[step], self.prec) for (k,v) in self.logs.items()}

        if step < 0:
            step = self.len_logs + step + 1 # + 1 hard coded for diffusion step
        print('step {}: {}'.format(step, step_stats))