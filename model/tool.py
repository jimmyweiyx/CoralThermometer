import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools
import os
import train
from tqdm import tqdm
class Tools():
    
    
    def DropNa(sel,Data):
        # load excel and deleted empty row according to 'type'
        LoadedFile = pd.read_excel(Data)
        DataWithoutNaGroup = LoadedFile.dropna(subset=['type'])
        return DataWithoutNaGroup

    def GetSSData(self, Data, XList, y='SST'):
        # manual split the data into 80% training set and 20% test set, according to groups
        X = Data[XList]
        y = Data['SST']
        groups = Data['type']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=groups)
        return X_train, y_train, X_test, y_test

    def cal_tag(self, y_true, y_pred):
        # calculation of performance metric
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2
    

    def Acombine(self, XList):
        # Acombine and Ccombine applied exhausted method on expore all posible proxy combinaiton of the XList
        lenth = len(XList)
        combine_list = []
        for i in range(lenth):
            number = i+1
            for args_combine in self.Ccombine(XList, number):
                combine_list.append(list(args_combine))
        return combine_list

    def Ccombine(self, list, C):
        _list = itertools.combinations(list,C)
        return _list

    
    def train(self,model,ParamGrid,SavingFile):
        # defined model with all dataset, all proxy combination, and weights
        # then put defined model into main training prograss
        XList = ['Mg/Ca', 'Sr/Ca', 'U/Ca', 'Li/Mg', 'B/Ca']
        model_train = train.Train()
        WorkDir = os.path.dirname(__file__)
        print('Training Start')
        for DataFile in tqdm(os.listdir(WorkDir + '/../dataTrain/'), desc='Data Files'):
            Path_2_DataFile = WorkDir + '/../dataTrain/' + DataFile
            for X in tqdm(self.Acombine(XList), desc='Proxy Combination'):
            
                for ifweighted in [True, False]:
                    model_train.Training(
                        Path_2_DataFile = Path_2_DataFile,
                        XList = X, 
                        Model = model, 
                        SavingFile = SavingFile, 
                        ParamGrid = ParamGrid, 
                        ifweighted=ifweighted,
                    )