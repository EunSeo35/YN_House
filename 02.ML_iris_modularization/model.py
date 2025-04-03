from sklearn.tree import DecisionTreeClassifier
from config import RANDOM_STATE, MODEL_TYPE


def get_model():
    if MODEL_TYPE == 'decision_tree':
        return DecisionTreeClassifier(random_state = RANDOM_STATE)
    else:
        raise ValueError(f"Unsuppored model type: {MODEL_TYPE}")
    
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


