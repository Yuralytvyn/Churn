from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
DATA_PATH = "./data/"
RATIO = [0.7, 0.15, 0.15]
SCALER = preprocessing.StandardScaler()
MODEL = LogisticRegression(max_iter=1000)