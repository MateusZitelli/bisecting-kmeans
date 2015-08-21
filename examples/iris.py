from kmeans.dataset import Dataset
from kmeans.bisectingKmeans import BisectingKmeans
from math import sqrt, ceil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class DataLoader():

    def __init__(self, filename, fields):
        def tryToFloat(value):
            try:
                return float(value)
            except ValueError:
                return value

        self.fields = fields
        loadedData = [tuple(map(tryToFloat, line.rstrip().split(',')))
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

if __name__ == "__main__":
    irisFields = [
        {'name': 'sepal length'},
        {'name': 'sepal width'},
        {'name': 'petal length'},
        {'name': 'petal width'},
        {'name': 'class', 'types': {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
        }}
    ]

    wineFields = [
        {'name': 'identifier'},
        {'name': 'alcohol'},
        {'name': 'malic acid'},
        {'name': 'ash'},
        {'name': 'alcalinity of ash'},
        {'name': 'magnesium'},
        {'name': 'total phenols'},
        {'name': 'flavanoids'},
        {'name': 'nonflavanoid phenols'},
        {'name': 'proanthocyanins'},
        {'name': 'color intesity'},
        {'name': 'hue'},
        {'name': 'OD280/OD315 of diluted wines'},
        {'name': 'proline'
         }
    ]

    hungarianFields = [
        {'name': 'age'},
        {'name': 'sex'},
        {'name': 'cp'},
        {'name': 'trestbps'},
        {'name': 'chol'},
        {'name': 'fbs'},
        {'name': 'restecg'},
        {'name': 'thalach'},
        {'name': 'exang'},
        {'name': 'oldpeak'},
        {'name': 'slope'},
        {'name': 'ca'},
        {'name': 'thal'},
        {'name': 'num'
         }
    ]

    loader = DataLoader('iris.data', irisFields)
    loader = DataLoader('wine.data', wineFields)
    loader = DataLoader('reprocessed.hungarian.data', hungarianFields)
    ds = Dataset(data=loader.data)
    bisection = BisectingKmeans(dataset=ds, k=3, trials=10, maxRounds=100)
    bisection.run()
    visualizer = MeansVisualizer(bisection.means, irisFields)
    visualizer.show()
