from sklearn.ensemble import ExtraTreesRegressor
import tool


# define ML regressor

model = ExtraTreesRegressor(
    n_estimators=1000
)


# define paramgrid

ParamGrid = {
    'train_model__max_features':[i+1 for i in range(5)],
    'train_model__bootstrap':[True,False],
    'train_model__ccp_alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10]
}

Tool = tool.Tools()

Tool.trainunweight(
    model,
    ParamGrid,
    SavingFile='ETR'
)
