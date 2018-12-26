import sys

class progressbar():
    def __init__(self, total_step, total_epoch, bar_len=60):
        self.bar_len = bar_len
        
        self.step = 1
        self.total_step = total_step
        self.epoch = 0
        self.total_epoch = total_epoch

    def add_epoch(self):
        self.epoch += 1
        self.step = 0

    def printf(self, text):
        filled_len = int(round(self.bar_len * self.step / float(self.total_step)))
        bar = '=' * filled_len + '-' * (self.bar_len - filled_len)

        sys.stdout.write(" Episode {%d/%d} [%s] (%d/%d) : %s\r " % (self.epoch, self.total_epoch, bar, self.step, self.total_step, text))
        sys.stdout.flush()

        self.step += 1