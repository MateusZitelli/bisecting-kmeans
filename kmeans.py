import random
from itertools import islice
from functools import partial


class WrongDimensionality(Exception):
    pass


class NoPoints(Exception):
    pass


class Dataset():
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
        if self.dimensions != len(point):
            if self.dimensions is not None:
                raise WrongDimensionality
            else:
                self.dimensions = len(point)
        self.data.append(point)

    def copy(self):
        return Dataset(data=self.data, normalized=True)

    def median(self):
        if self.dimensions is None:
            raise NoPoints("There is no data points for median acquisition.")

        return [sum(dimension)/len(self.data) for dimension in zip(*self.data)]


class Mean():
    def __init__(self, id, dimensions):
        self.id = id
        self.dimensions = dimensions
        self.position = [random.random() for i in range(dimensions)]
        self.coveredDataset = None
        self.nextDataset = Dataset(normalized=True, data=[])

    def update(self):
        if self.coveredDataset == self.nextDataset:
            self.nextDataset = Dataset(normalized=True, data=[])
            return False
        self.coveredDataset = self.nextDataset.copy()
        self.position = self.coveredDataset.median()
        return True

    def clear(self):
        self.nextDataset = Dataset(normalized=True, data=[])

    def cover(self, point):
        self.nextDataset.append(point)

    def __repr__(self):
        return "<Mean id: %i position:%s dataset:%s>" % (
            self.id, str(self.position), self.coveredDataset.__repr__())

    def distanceSqrd(self, point):
        def dist(v1, v2):
            return sum([(j - v2[i]) ** 2 for i, j in enumerate(v1)])
        if len(point) != len(self.position):
            raise WrongDimensionality()
        return dist(point, self.position)


class KmeansSolution():
    def __init__(self, dataset, k, rounds):
        self.k = k
        self.dataset = dataset
        self.rounds = rounds
        self.means = [Mean(i, dataset.dimensions) for i in range(k)]
        self.solve()

    def __repr__(self):
        return str(self.means)

    def solve(self):
        for r in range(self.rounds):
            for point in self.dataset:
                nearstMean = min(self.means,
                                 key=lambda m: m.distanceSqrd(point))
                nearstMean.cover(point)
            hasChanges = False
            for mean in self.means:
                try:
                    hasChanges = hasChanges or mean.update()
                except NoPoints:
                    pass
                mean.clear()

            if not hasChanges:
                return


class Kmeans():
    def __init__(self, dataset, k, trials, rounds):
        self.dataset = dataset
        self.k = k
        self.trials = trials
        self.rounds = rounds

    def run(self):
        self.solutions = [KmeansSolution(self.dataset, self.k, self.rounds)
                          for t in range(self.trials)]

    def showResults(self):
        for solution in self.solutions:
            print(solution)

if __name__ == "__main__":
    ds = Dataset([[0, 0], [1, 1]])
    k = Kmeans(ds, 2, 5, 3)
    k.run()
    k.showResults()
