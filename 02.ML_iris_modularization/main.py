from data_loader import load_data
from model import get_model, train_model
from evaluation import evaluate_model
from validation import cross_validation
from config import VERBOSE


# 1. 데이터 로드 
X_train, X_test, y_train, y_test = load_data()

# 2. 모델 로드 및 모델 학습 
model = get_model()
model = train_model(model, X_train, y_train)

# 3. 검증 
cross_validation(model, X_train, y_train)

# 4. 모델 정확성 평가 
accuracy = evaluate_model(model, X_test, y_test)
if VERBOSE:
    print('최종 예측 정확도: {0:.3f}'.format(accuracy))