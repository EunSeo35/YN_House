from sklearn.metrics import accuracy_score

def evaluate_model(model,X_test,y_test):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    return acc 

