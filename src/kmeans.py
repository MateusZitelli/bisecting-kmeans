import random
from .exceptions import NoPoints, WrongDimensionality
from .dataset import Dataset


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
