from kmeans.dataset import Dataset
from kmeans.bisectingKmeans import BisectingKmeans
from utils import DataLoader, MeansVisualizer, KneeFinder

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

loader = DataLoader('iris.data', irisFields)
ds = Dataset(data=loader.data)
kf = KneeFinder(dataset=ds, krange=[1, 10], trials=50, maxRounds=100)
kf.run()
kf.show()
bisection = BisectingKmeans(dataset=ds, k=2, trials=30, maxRounds=100)
bisection.run()
visualizer = MeansVisualizer(bisection.means, irisFields)
visualizer.show()
