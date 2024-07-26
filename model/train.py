import tool
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import os
import joblib


class Train():
    # This is main traing prograss
    # which call model to fit data with given situation
    def Training(self, Path_2_DataFile, XList, Model, SavingFile, ParamGrid, ifweighted=False):
        # Path_2_DataFile refers to reletive or absolute path of DataFile
        # XList refers to Proxy Combination, accpeted cell or list as ['Sr/Ca','Li/Mg']
        # Model refers to regression model functions from sklearn
        # SavingFile refers to the name of savemodel
        # Paramgrid refers to predefined hyperparameter grid using in grid search
        # ifweighted refers to whether using class weights to balance the dataset.
        WorkDir = os.path.dirname(__file__)
        DataFileName = Path_2_DataFile.split('/')[-1].split('.')[0]
        Elements = '.'.join('_'.join(XList).split('/'))
        SavedDirName = f'{WorkDir}/../SavedFile/{DataFileName}/{Elements}/'
        Tool = tool.Tools()
        Data = Tool.DropNa(Path_2_DataFile)
        groups = Data['type']


        pl = Pipeline([('Sacler', StandardScaler()), ('train_model', Model)])
        # split dataset into 20% test set and 80% training set
        X_Calibration, y_Calibration, X_Validation, y_Validation = Tool.GetSSData(Data, XList, 'SST')

        # get training set index for balancing
        index_train = X_Calibration.index

        # defined gridsearch scheme, with 10-fold cross-validation
        training = GridSearchCV(pl, 
                                param_grid=ParamGrid, 
                                cv=10,
                                refit=True, 
                                n_jobs=-1)
        # set weight and non-weight model, if weights is applied, using 'balanced' scheme to calculate the samples weight with training index
        if ifweighted:
            sample_weights = compute_sample_weight(class_weight='balanced', y=groups.iloc[index_train])
            # add weight to lossfunc
            kwargs = {pl.steps[-1][0] + '__sample_weight': sample_weights}
            # call model training
            training.fit(X_Calibration, y_Calibration, **kwargs)
        else:
            # if weight not set, directly call model training
            training.fit(X_Calibration, y_Calibration)


        if os.path.isdir(SavedDirName) == False:
            os.makedirs(SavedDirName)
            
        # saving model file 
        if ifweighted:
            joblib.dump(training, SavedDirName + SavingFile + 'weighted.m')
        else:
            joblib.dump(training, SavedDirName + SavingFile + '.m')
