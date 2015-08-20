import random
from exceptions import NoPoints, WrongDimensionality
from dataset import Dataset
from math import log


class Mean():
    """
    A mean of the k-means.
    """
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
        def distSqrd(v1, v2):
            return sum([(j - v2[i]) ** 2 for i, j in enumerate(v1)])
        if len(point) != len(self.position):
            raise WrongDimensionality()
        return distSqrd(point, self.position)

    def totalSquaredError(self):
        squaredDists = [self.distanceSqrd(point) for point in self.coveredDataset]
        return sum(squaredDists)



class KmeansSolution():
    """
    Represents a k-means execution.
    """
    def __init__(self, dataset, k, maxRounds):
        self.k = k
        self.dataset = dataset
        self.maxRounds = maxRounds
        self.means = [Mean(i, dataset.dimensions) for i in range(k)]
        self.meanSquaredError = None
        self.solve()

    def __repr__(self):
        meansString = "\n  ".join([str(mean) for mean in self.means])
        return "<Solution mean squared error: %f means:\n  %s>" % (
            self.meanSquaredError, meansString)

    def solve(self):
        for r in range(self.maxRounds):
            hasChanges = False
            for point in self.dataset:
                nearstMean = min(self.means,
                                 key=lambda m: m.distanceSqrd(point))
                nearstMean.cover(point)

            for mean in self.means:
                try:
                    hasChanges = hasChanges or mean.update()
                except NoPoints:
                    pass
                mean.clear()

            if not hasChanges:
                break
        self.setMeanSquaredError()

    def setMeanSquaredError(self):
        totalSquaredError = sum([mean.totalSquaredError() for mean in self.means])
        self.meanSquaredError = totalSquaredError / len(self.dataset)



class Kmeans():
    """
    Kmeans main class where the algorithm will be exectued many times with
    different random initial conditions.
    """

    def __init__(self, dataset, k, trials, maxRounds):
        """
        dataset - The aim dataset
        k - The number of means
        trials - How many times the algorithm will be executed
        maxRounds - The maximum number of iterations before stop each execution
        """
        self.dataset = dataset
        self.k = k
        self.trials = trials
        self.maxRounds = maxRounds

    def run(self):
        self.solutions = [KmeansSolution(self.dataset, self.k, self.maxRounds)
                          for t in range(self.trials)]
        self.solutions.sort(key=lambda s: s.meanSquaredError)

    def showResults(self):
        for solution in self.solutions:
            print(solution)

if __name__ == "__main__":
    ds = Dataset(data=[[0, 0], [1, 1]])
    k = Kmeans(dataset=ds, k=2, trials=5, maxRounds=3)
    k.run()
    k.showResults()
