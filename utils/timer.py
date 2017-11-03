from datetime import timedelta
from time import time

#high level decoretor
def step_time(time_name = 'Step'):
    def inner(func):
        def wrapper(*args, **kargs):
            start_time = time()
            rv = func(*args, **kargs)
            print(time_name + ' time: {:.3f}s'.format(time() - start_time))
            return rv
        return wrapper
    return inner   

class Timer(object):
    '''
    A simple timer to display the average and remaining time
    '''
    def __init__(self):
        self.init_time = time()
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.remain_time = 0.
        
    def tic(self):
        '''
        start individual operation timing
        '''
        self.start_time = time()

    def toc(self, average=True, show = False):
        '''
        end individual operation timing
        '''
        self.diff = time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            if show:
                print('Average detecting time: {:.3f}s'.format(self.average_time))
            return self.average_time
        else:
            if show:
                print('Delta time: {:.3f}s'.format(self.diff))
            return self.diff

    def remain(self, iters, max_iters):
        if iters == 0:
            self.remain_time = 0
        else:
            self.remain_time = (time()-self.init_time)*(max_iters-iters)/iters
        return str(timedelta(seconds=int(self.remain_time)))
