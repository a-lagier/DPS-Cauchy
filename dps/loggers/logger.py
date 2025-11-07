class Logger():

    def __init__(self, keys: list, prec: int = 5):
        self.logs = {k:[] for k in keys}
        self.prec = prec
        self.len_logs = 0
    
    def update_stats(self, **kwargs):
        for metric in kwargs:
            self.logs[metric].append(kwargs[metric])
        self.len_logs += 1
    
    def write_step(self):
        step_stats = {k: round(v[-1], self.prec) for (k,v) in self.logs.items()}

        print('step {}: {}'.format('result', step_stats))