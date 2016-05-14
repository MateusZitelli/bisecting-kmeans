from kmeans.dataset import Dataset
from kmeans.bisectingKmeans import BisectingKmeans
from utils import DataLoader, MeansVisualizer, KneeFinder

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
    {'name': 'proline'}
]

loader = DataLoader('wine.data', wineFields)
ds = Dataset(data=loader.data)
# kf = KneeFinder(dataset=ds, krange=[1, 10], trials=100, maxRounds=100)
# kf.run()
# kf.show()
bisection = BisectingKmeans(dataset=ds, k=4, trials=99, maxRounds=100, key=lambda x: x[:2])
bisection.run()
visualizer = MeansVisualizer(bisection.means, wineFields)
visualizer.show()
