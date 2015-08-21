from kmeans.dataset import Dataset
from kmeans.bisectingKmeans import BisectingKmeans
from utils import DataLoader, MeansVisualizer

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
    {'name': 'num'}
]

loader = DataLoader('reprocessed.hungarian.data', hungarianFields)
print(loader.data)
ds = Dataset(data=loader.data)
bisection = BisectingKmeans(dataset=ds, k=3, trials=99, maxRounds=100)
bisection.run()
visualizer = MeansVisualizer(bisection.means, hungarianFields)
visualizer.show()
