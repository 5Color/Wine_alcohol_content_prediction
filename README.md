# Wine_alcohol_content_prediction

와인 알코올 도수 예측을 하기위한 회귀모델을 만들어보았다

# 사용 데이터:
우선 사용된 데이터는 WineQT.csv으로 1143 rows × 13 columns이다.

# import
우선 첫번째로 [pandas, sklearn의 train_test_split과 linear_model인 LinearRegression, lightgbm의 LGBMRegressor, ensemble의 RandomForestRegressor]라이브러리를 불러온다.

# 데이터 로드
pandas의 read_csv함수를 이용해 WineQT.csv데이터를 불러오고 data변수에 저장한다

# 전처리
이제 독립변수(input) x와 종속변수(target) y컬럼을 설정해주어야하는데 
x = data[['fixed acidity','volatile acidity','citric acid',
          'residual sugar','chlorides','free sulfur dioxide',
          'total sulfur dioxide','density','pH','sulphates','quality']]
y = data['alcohol'] 
이렇게 하여 종속변수y는 alcohol컬럼으로 지정하여 회귀문제로 만든다.
y.shape를 통해 확인했더니 1차원에 1143개의 데이터 (1143,) 출력이 되었다

두번째로 데이터셋을 train_test_split함수로 분리해준다. 
train_input, test_input, train_target, test_target으로 변수를 설정해서 분리해주고
train_input.shape를 통해 (914, 11) 이 독립변수 데이터가 2차원에 914개의 데이터, 11개의 컬럼으로 이루어져 있다는것을 확인할 수있다

# 모델 학습 및 결과

## lightGBM 모델
lightgbm의 LGBMRegressor()을 사용해 객체 생성, 데이터 fitting하고, 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error를 불러와
평가지표를 확인한다. 결과는 놀라웠다.

MAE :  0.3184846742536364
MSE :  0.1993415732824
R^2 결정계수 :  0.7575681050572329
RMSE :  0.44647684518057595

2번째로 이 데이터에서 성능이 좋았다.
r^2 상관계수가 높은걸 보아 데이터를 잘 이해하고 있다는 증표다.

## 선형회귀(LinearRegression) 모델
LinearRegression()을 사용해 reg변수에 객체 생성, 데이터 fitting하고, 평가지표 확인해본 결과는 그닥 좋진 않았다..

MAE :  0.4275155169591243
MSE :  0.3257906738394674
R^2 결정계수 :  0.6048168079480349
RMSE :  0.5707807581194967

## 랜덤포레스트(RandomForest) 모델
from sklearn.ensemble import RandomForestRegressor 호출, RandomForestRegressor()을 사용해 randomforest변수에 객체 생성,
데이터 fitting하고, 평가지표 확인해본 결과..

MAE :  0.35594564531780737
MSE :  0.24366423895631192
R^2 결정계수 :  0.6374653126471075
RMSE :  0.4936235802271929

무난하다 ㅋㅋ

## XGBoost 모델
from xgboost import XGBRegressor 호출, XGBRegressor()을 사용해 xgb변수에 객체 생성, 데이터 fitting하고, 평가지표 확인 해본결과..!

MAE :  0.3060099855781122
MSE :  0.2081237271535875
R^2 결정계수 :  0.748662956705184
RMSE :  0.45620579473915884

은근 성능이 잘나왔다. 랜덤포레스트 모델보단 나은것 같다.


##  마지막으로 앙상블스태킹(ensemble_StackingRegressor) 모델
이 학습 알고리즘은 처음써본다. 앙상블 방법중에 가장 독특한것 같아 선정했고, 성능도 잘 나온다길래 써보았다
from sklearn.ensemble import StackingRegressor 호출, 우선 첫번째 stack단계에서 학습에 사용할 모델은 LinearRegression(), RandomForestRegressor(n_estimators=200), LGBMRegressor(n_estimators=300)이다 이것을 base_learners변수에 저장한다.
여기서 n_estimators매개변수는 해당 모델을 직렬/병렬 모델로 몇개를 사용할것이냐를 따지는것이다.

stacking = StackingRegressor(estimators=base_learners)이렇게 코드를 짜서 객체를 생성해주고
데이터를 fitting해준다. 학습은 앙상블 스태킹기반이라 조금 느리다. 평자지표를 확인 해본 결과는...!!!

MAE :  0.31594546837174187
MSE :  0.18784150519917675
R^2 결정계수 :  0.778698716353184
RMSE :  0.4334068587357343

와 모델중에서 이게 제일 성능이 좋다. 1등임!
살짝 아쉬웠던점은 stacking.score(train_input, train_target) 결과가 0.9572395251594029,
stacking.score(test_input, test_target) 결과가 0.8173739298407945 로
살짝 과적합 되어있다는건데 이걸로 만족한다!

