from units import msec

class TimeKeeper:
    def __init__(self, timeunit=1.0*msec):
        self.tick = 1
        self.timeunit = timeunit

    def dt(self):
        return self.timeunit
    
    def step_forward(self):
        self.tick += 1

class TimeKeeperIterator(TimeKeeper):
    def __init__(self, timeunit=1.0*msec):
        super(TimeKeeperIterator, self).__init__(timeunit=timeunit)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.step_forward()
        return self.tick
