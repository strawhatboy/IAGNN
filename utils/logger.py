
import time
import sys
import logging

def red_str(s, tofile=False):
    s = str(s)
    if tofile:
        # s = f'**{s}**'
        pass
    else:
        s = f'\033[1;31;40m{s}\033[0m'
    return s


class Logger:
    def __init__(self, fn='xx.log', verbose=1):
        self.pre_time = time.time()
        self.fn = fn
        self.verbose = verbose

    def __str__(self):
        return self.fn

    def log(self, s='', end='\n', red=False):
        s = str(s)
        if self.verbose == 1:
            p = red_str(s) if red else s
            print(p, end=end)
        # elif self.verbose == 2:
        #     p = red_str(s, tofile=True) if red else s
        #     print(p, end=end)
        # now_time = time.time()
        # s = s + end
        # if now_time - self.pre_time > 30 * 60:
        #     s = get_time_str() + '\n' + s
        #     self.pre_time = now_time
        with open(self.fn, 'a') as f:
            fs = red_str(s, tofile=True) if red else s
            f.write(fs)
        sys.stdout.flush()


# def init_logger(conf):
#     logger = logging.getLogger(__name__)
#     logger.setLevel(level=logging.INFO)
#     handler = logging.FileHandler("./logs/log_{}_{}_{}.txt".format(conf['recommender'], \
#                                                                    conf['epoch'], conf['comment']))
#     handler.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     console = logging.StreamHandler()
#     console.setLevel(logging.INFO)
#
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.addHandler(console)
#     return logger