from sklearn.svm import SVR
import tool


# define ML regressor
model = SVR(kernel='rbf')

# define paramgrid
ParamGrid = {
    'train_model__gamma':['scale', 'auto'],
    'train_model__C':[1,2,3,4,5],
    'train_model__epsilon':[i / 100 for i in range(1, 50)]
}

Tool = tool.Tools()
Tool.train(
    model,
    ParamGrid,
    SavingFile='SVR'
)