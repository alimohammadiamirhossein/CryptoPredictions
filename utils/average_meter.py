class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def get_average(self):
        assert not (self.count == 0 and self.sum != 0)
        if self.count == 0:
            return 0
        return self.sum / self.count
