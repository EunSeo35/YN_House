from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

