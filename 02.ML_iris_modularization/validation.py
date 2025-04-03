from sklearn.model_selection import cross_val_score
import numpy as np

def cross_validation(model, X_train, y_train, cv = 4):
    scores  = cross_val_score(model, X_train, y_train, scoring = 'accuracy', cv=4)
    print('교차 검증별 정확도', np.round((scores),2))
    print('평균 검증 정확도', np.round(np.mean(scores),2))
    return scores