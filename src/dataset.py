from itertools import islice
from functools import partial
from .exceptions import NoPoints, WrongDimensionality


class Dataset():
    """ Data set representation with automatic data normalization. """
    def __init__(self, data=[], normalized=False):
        def normalizeValue(ceil, floor, value):
            return (value - floor) / (ceil - floor)

        def normalize(data):
            minimum = min(map(min, data))
            maximum = max(map(max, data))
            self.dataCeil = maximum
            self.dataFloor = minimum
            normalizer = partial(normalizeValue, maximum, minimum)
            return [list(map(normalizer, v)) for v in data]

        self.data = normalize(data) if not normalized else data

        if(len(self.data) > 0):
            self.dimensions = len(self.data[0])
        else:
            self.dimensions = None

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current == len(self):
            raise StopIteration
        self.current += 1
        return self.data[self.current - 1]

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.dimensions != other.dimensions:
            return False
        self.data.sort()
        other.data.sort()
        return other.data == self.data

    def __getitem__(self, key):
        if isinstance(key, int) and key >= 0:
            return self.data[key]
        elif isinstance(key, slice):
            return islice(self.data, key.start, key.stop, key.step)
        else:
            raise KeyError("Key must be non-negative integer or slice, not {}"
                           .format(key))

    def __repr__(self):
        return "<Dataset points:%s>" % (self.data)

    def append(self, point):
        """
        Adds a point to the dataset if the dimensionality of the point
        mathces the dataset dimensionality, in case of unset dimensionality it
        will be defined based on the length of the point vector. Raise an
        WrongDimensionality error in case of incompatible data
        """

        if self.dimensions != len(point):
            if self.dimensions is not None:
                raise WrongDimensionality
            else:
                self.dimensions = len(point)
        self.data.append(point)

    def copy(self):
        return Dataset(data=self.data, normalized=True)

    def median(self):
        """
        Return the median of the data set, if there is no data points a NoPoints
        error will be raised.
        """

        if self.dimensions is None:
            raise NoPoints("There is no data points for median acquisition.")

        return [sum(dimension)/len(self.data) for dimension in zip(*self.data)]
