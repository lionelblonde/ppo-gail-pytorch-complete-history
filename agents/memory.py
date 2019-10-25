import numpy as np


class RingBuffer(object):

    def __init__(self, maxlen, shape, dtype='float32'):
        """Ring buffer implementation"""
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape, dtype=dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v, offset=0):
        """`offset` is used to keep the demos forever in the buffer (never flushed out by FIFO)"""
        global debug
        if self.length < self.maxlen:
            # We have space, simply increase the length
            self.length += 1
            self.data[offset + (self.start + self.length - 1)
                      % (self.maxlen - offset)] = v

        elif self.length == self.maxlen:
            # No space, "remove" the first item
            self.start = (self.start + 1) % (self.maxlen - offset)
            self.data[offset + (self.start + self.length - offset - 1)
                      % (self.maxlen - offset)] = v
        else:
            # This should never happen
            raise RuntimeError()
