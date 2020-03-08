from collections import OrderedDict

class OnChipBuffer(): # LRU
    def __init__(self, size):
        self.size = size
        self.buffer = OrderedDict()
        self.maximal_usage = 0
        
    def put(self, key, value):
        if key in self.buffer:
            self.buffer.pop(key)

        if self.size == len(self.buffer):
            self.buffer.popitem(last=False)

        self.buffer.update({key: value})
        self.maximal_usage = max(len(self.buffer), self.maximal_usage)

    def get(self, key):
        value = self.buffer.get(key)
        if not value:
            return None
        self.buffer.pop(key)
        self.buffer.update({key: value})
        return value
    
    def pop(self, key):
        self.buffer.pop(key)
    
    def __str__(self):
        return str(self.buffer)