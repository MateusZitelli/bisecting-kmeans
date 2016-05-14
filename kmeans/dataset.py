from itertools import islice
from kmeans.exceptions import NoPoints, WrongDimensionality


class Dataset():
    """ Data set representation with automatic data normalization. """
    def __init__(self, data=[], normalized=False):
        def normalizeValue(ceil, floor, value):
            offset = (ceil - floor)
            if offset == 0:
                return value - floor
            return (value - floor) / offset

        def normalize(data):
            zipped = list(zip(*data))
            self.minimuns = list(map(min, zipped))
            self.maximuns = list(map(max, zipped))
            return [[normalizeValue(self.maximuns[i], self.minimuns[i], d)
                     for i, d in enumerate(v)]
                    for v in data]

        self.minimuns = None
        self.maximuns = None
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
            raise KeyError("Key must be non-negative integer or slice, not {}".format(key))

    def __repr__(self):
        return "<Dataset points:%s>" % (self.data)

    def genUnnormalized(self, limits):
        maximuns, minimuns = limits

        def unnormalizeValue(ceil, floor, value):
            return value * (ceil - floor) + floor

        unnormalizedData =  [[
            unnormalizeValue(maximuns[i], minimuns[i], d)
            for i, d in enumerate(v)]
            for v in self.data]
        return unnormalizedData

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
        ds = Dataset(data=self.data, normalized=True)
        if self.maximuns is not None and self.minimuns is not None:
            ds.setLimits(self.maximuns, self.minimuns)
        return ds

    def median(self):
        """
        Return the median of the data set, if there is no data points a NoPoints
        error will be raised.
        """

        if self.dimensions is None:
            raise NoPoints("There is no data points for median acquisition.")

        return [sum(dimension)/len(self.data) for dimension in zip(*self.data)]

    def setLimits(self, maximuns, minimuns):
        """
        Define maximuns and minimuns
        """
        self.minimuns = minimuns
        self.maximuns = maximuns
