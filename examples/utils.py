import re
from math import sqrt, ceil

import matplotlib.pyplot as plt
from kmeans.bisectingKmeans import BisectingKmeans
from kmeans.kmeans import Kmeans


class DataLoader():
    def __init__(self, filename, fields):
        def tryToFloat(value):
            try:
                return float(value)
            except ValueError:
                return value

        self.fields = fields
        loadedData = [tuple(map(tryToFloat, re.split('[,\s]', line.rstrip())))
                      for line in open(filename, 'r').readlines()][:-1]

        dataZiped = list(zip(*loadedData))

        for i, field in enumerate(fields):
            if 'types' in field:
                try:
                    dataZiped[i] = [field['types'][v] for v in dataZiped[i]]
                except KeyError as e:
                    print(e)
                    raise Exception("The field structure don't fit the data.")

        self.data = list(zip(*dataZiped))


class MeansVisualizer():
    def __init__(self, means, fields):
        self.means = means
        self.dimensions = self.means[0].coveredDataset.dimensions
        self.numberOfPlots = int(ceil(sqrt(self.dimensions / 2)))
        self.fields = fields
        self.fig, self.axs = plt.subplots(nrows=self.numberOfPlots,
                                          ncols=self.numberOfPlots)

    def setData(self, xcord, ycord):
        self.data = [mean.coveredDataset for mean in self.means]

    def plot(self, idx, idy, datax, datay, dataIds):
        ax = self.axs[idy][idx]
        ax.plot(datax, datay, 'o')
        ax.set_xlabel(self.fields[dataIds[0]]['name'])
        ax.set_ylabel(self.fields[dataIds[1]]['name'])

    def show(self):
        pairDimensions = self.dimensions // 2

        for mean in self.means:
            if self.dimensions % 2 == 1:
                idx = self.dimensions // 2 % self.numberOfPlots
                idy = self.dimensions // 2 // self.numberOfPlots
                dataIds = [self.dimensions - 1, 0]
                datax = [d[dataIds[0]] for d in mean.coveredDataset]
                datay = [d[dataIds[1]] for d in mean.coveredDataset]
                self.plot(idx, idy, datax, datay, dataIds)

            for i in range(0, pairDimensions):
                idx = i % self.numberOfPlots
                idy = i // self.numberOfPlots
                dataIds = [i * 2, i * 2 + 1]
                datax = [d[dataIds[0]] for d in mean.coveredDataset]
                datay = [d[dataIds[1]] for d in mean.coveredDataset]
                self.plot(idx, idy, datax, datay, dataIds)

        plt.show()


class KneeFinder():
    def __init__(self, dataset, krange, trials, maxRounds):
        self.dataset = dataset
        self.krange = range(*krange)
        self.trials = trials
        self.maxRounds = maxRounds

    def run(self):
        self.bisecting = [BisectingKmeans(dataset=self.dataset, k=i,
                                          trials=self.trials,
                                          maxRounds=self.maxRounds)
                          for i in self.krange]
        self.normal = [Kmeans(dataset=self.dataset, k=i, trials=self.trials,
                              maxRounds=self.maxRounds) for i in self.krange]

        for i, j in enumerate(self.krange):
            self.bisecting[i].run()
            self.normal[i].run()

    def show(self):
        bisectingErrors = [b.meanSquaredError for b in self.bisecting]
        normalErrors = [b.getBestSolution().meanSquaredError
                        for b in self.normal]
        plt.plot(self.krange, bisectingErrors)
        plt.plot(self.krange, normalErrors)
        plt.show()
