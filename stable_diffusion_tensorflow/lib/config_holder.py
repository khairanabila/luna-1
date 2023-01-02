import copy

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class cfg_unique_holder(object):
    def __init__(self):
        self.cfg = None
        self.code = set()
    def save_cfg(self, cfg):
        self.cfg = copy.deepcopy(cfg)
    def add_code(self, code):
        self.code.add(code)
