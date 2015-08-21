from kmeans.kmeans import Kmeans, Mean
from kmeans.dataset import Dataset


class BisectingKmeans():
    def __init__(self, dataset, k, trials, maxRounds):
        """
        dataset - The aim dataset
        k - The number of means
        trials - How many times the k-means will be executed in each bisection
        maxRounds - The maximum number of iterations before stop each execution
            of the k-means.
        """
        self.dataset = dataset
        self.k = k
        self.trials = trials
        self.maxRounds = maxRounds
        self.means = []
        self.meanSquaredError = None

    def setMeanSquaredError(self):
        totalSquaredError = sum([mean.getTotalSquaredError()
                                 for mean in self.means])
        self.meanSquaredError = totalSquaredError / len(self.dataset)

    def run(self):
        worstCluster = self.dataset

        while len(self.means) < self.k:
            if isinstance(worstCluster, Dataset):
                worstDataset = worstCluster
            elif isinstance(worstCluster, Mean):
                worstDataset = worstCluster.coveredDataset

            bisection = Kmeans(dataset=worstDataset, k=2, trials=self.trials,
                               maxRounds=self.trials)
            bisection.run()
            bisectionSolution = bisection.getBestSolution()

            self.means += bisectionSolution.means

            worstCluster = bisectionSolution.getWorstMean()

            # if the number of means is not enouth remove the worst cluster
            # found to bisect it in the next iteration.
            if len(self.means) < self.k:
                self.means.remove(worstCluster)
        self.setMeanSquaredError()

    def showResults(self):
        print('\n\n'.join([str(mean) for mean in self.means]))

if __name__ == "__main__":
    ds = Dataset(data=[[0, 0], [1, 1], [0.9, 0.9], [0.5, 0.5]])
    bisection = BisectingKmeans(dataset=ds, k=4, trials=20, maxRounds=3)
    bisection.run()
    bisection.showResults()
