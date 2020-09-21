import numpy as np
from collections import OrderedDict


def mat_argmax(m_A):
    """ returns tuple with indices of max entry of matrix m_A"""

    num_cols = m_A.shape[1]
    assert m_A.ndim == 2

    ind = np.argmax(m_A)

    row = ind // num_cols
    col = ind % num_cols
    return (row, col)


def mat_argmin(m_A):
    """ returns tuple with indices of min entry of matrix m_A"""

    num_cols = m_A.shape[1]
    assert m_A.ndim == 2

    ind = np.argmin(m_A)

    row = ind // num_cols
    col = ind % num_cols
    return (row, col)


def print_time(start_time, end_time):
    td = end_time - start_time
    hours = td.seconds // 3600
    reminder = td.seconds % 3600
    minutes = reminder // 60
    seconds = (td.seconds - hours * 3600 -
               minutes * 60) + td.microseconds / 1e6
    time_str = ""
    if td.days:
        time_str = "%d days, " % td.days
    if hours:
        time_str = time_str + "%d hours, " % hours
    if minutes:
        time_str = time_str + "%d minutes, " % minutes
    if time_str:
        time_str = time_str + "and "

    time_str = time_str + "%.3f seconds" % seconds
    #set_trace()
    print("Elapsed time = ", time_str)


def empty_array(shape):
    return np.full(shape, fill_value=None, dtype=float)


def project_to_interval(x, a, b):
    assert a <= b
    return np.max([np.min([x, b]), a])


def watt_to_dbm(array):
    assert (array > 0).all()

    return 10 * np.log10(array) + 30


def dbm_to_watt(array):
    return 10**((array - 30) / 10)


def natural_to_dB(array):  # array is power gain
    return 10 * np.log10(array / 10)


class FifoUniqueQueue():
    """FIFO Queue that does not push a new element if it is already in the queue. Pushing an element already in the queue does not change the order of the queue.
    It seems possible to implement this alternatively as a simple list.
"""
    def __init__(self):
        self._dict = OrderedDict()

    def put(self, key):
        self._dict[key] = 0  # dummy value, for future usage

    def get(self):
        # Returns oldest item
        return self._dict.popitem(last=False)[0]

    def empty(self):
        return len(self._dict) == 0