import numpy as np
import pandas as pd
import sklearn
# If you import here, you can use it.
from sklearn.linear_model import LogisticRegression, HuberRegressor, LinearRegression,Ridge,Perceptron
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor

class MakeMLModel():

    def _choose_data(self, data, columns):
        """
        data (DataFrame): input data
        columns (list): Column number or column name of the specified column
        """
        if all([type(x) == int for x in columns]):
            choose_data = data.iloc[:, columns]
        elif all([type(x) == str for x in columns]):
            choose_data = data.loc[:, columns]
        else:
            raise ValueError("Use same type or correct type")

        return np.array(choose_data)

    def _algorithms_align_same_shape(self, len_mid, len_obj, algorithms):
        """
        Regardless of what type of algorithm comes in, change to pattern 5.
        1: 'GaussianProcessRegressor'
        2:  ['GaussianProcessRegressor',{'n_estimators':1}]
        3 : [['GaussianProcessRegressor'], ['GaussianProcessRegressor']]
        4 : [['GaussianProcessRegressor',{'n_estimators':1}], ['GaussianProcessRegressor', {}]]
        5 : [[['GaussianProcessRegressor',{'n_estimators':1}] for x in range(len(mid))], [['GaussianProcessRegressor',{'n_estimators7:1}] for x in range(len(obj))]]
        If there is no mid,
        1 : 'GaussianProcessRegressor'
        2 : ['GaussianProcessRegressor', {}]
        3 : [[], ['GaussianProcessRegressor']]
        4 : [[], ['GaussianProcessRegressor', {}]]
        5 : [[], [['GaussianProcessRegressor',{'n_estimators':1}] for x in range(len(obj))]]
        """
        if len_mid>=1:
            # If there is mid,
            # Judgment based on the type of np.array
            if np.array(algorithms).shape==() and np.array(algorithms[1]).shape==():
                # For pattern 1
                algorithms_ =[[[algorithms,{}] for x in range(len_mid)], [[algorithms,{}] for y in range(len_obj)]]
            elif np.array(algorithms).shape==(2,) and np.array(algorithms[1]).shape==():
                # For pattern 2
                algorithms_ =[[[algorithms[0],algorithms[1]] for x in range(len_mid)], [[algorithms[0],algorithms[1]] for y in range(len_obj)]]
            elif np.array(algorithms).shape==(2,1) and np.array(algorithms[1]).shape==(1,):
                # For pattern 3
                algorithms_ =[[[algorithms[0][0],{}] for x in range(len_mid)], [[algorithms[1][0],{}] for y in range(len_obj)]]
            elif np.array(algorithms).shape==(2,2) and np.array(algorithms[1]).shape==(2,):
                # For pattern 4
                algorithms_ =[[[algorithms[0][0],algorithms[0][1]] for x in range(len_mid)], [[algorithms[1][0],algorithms[1][1]] for y in range(len_obj)]]
            elif np.array(algorithms).shape==(2,) and len(np.array(algorithms[1]).shape)==2:
                # For pattern 5
                algorithms_ = algorithms 
            elif (np.array(algorithms).shape==(2,) and np.array(algorithms[1]).shape==(1,)) \
                    or (np.array(algorithms).shape==(2,) and np.array(algorithms[1]).shape==(2,)):
                raise ValueError("algorithms or len_mid is incorrect")
            else:
                raise ValueError("algorithms shape is incorrect")
                
        elif len_mid==0 or len_mid==None or len_mid==False:
            # If there is no mid
            # Judgment based on the type of np.array
            if np.array(algorithms).shape==() and np.array(algorithms[1]).shape==():
                # For pattern 1
                algorithms_ =[[], [[algorithms,{}] for y in range(len_obj)]]
            elif np.array(algorithms).shape==(2,) and np.array(algorithms[1]).shape==():
                # For pattern 2
                algorithms_ =[[], [[algorithms[0],algorithms[1]] for y in range(len_obj)]]
            elif (np.array(algorithms).shape==(2,1) and np.array(algorithms[1]).shape==(1,)) \
                        or (np.array(algorithms).shape==(2,) and np.array(algorithms[1]).shape==(1,)):
                # For pattern 3
                algorithms_ =[[], [[algorithms[1][0],{}] for y in range(len_obj)]]             
            elif (np.array(algorithms).shape==(2,2) and np.array(algorithms[1]).shape==(2,)) \
                        or (np.array(algorithms).shape==(2,) and np.array(algorithms[1]).shape==(2,)):
                # For pattern 4
                algorithms_ =[[], [[algorithms[1][0],algorithms[1][1]] for y in range(len_obj)]]            
                
            elif np.array(algorithms).shape==(2,) and len(np.array(algorithms[1]).shape)==2:
                # For pattern 5
                algorithms_ = [[], algorithms[1]]
            else:
                raise ValueError("algorithms shape is incorrect")
        else:
            raise ValueError('len_mid is incorrect')
        
        return algorithms_

    def fit(self, data, exp, obj, mid=None, 
                algorithms=[["GaussianProcessRegressor",{}],["GaussianProcessRegressor",{}]],
                  scale=True, ):
        """
        Fitting for multiple objective variable data.
        Make models as follows; [[exp1→mid1, exp2→mid2,...],[mid1→obj1,mid2→obj2,...]]
        or mid is None; [[],[exp1→obj1,exp2→obj2,...]]

        data (DataFrame): Input data. Make sure that the columns are in the order exp, mid, obj.
        exp (list): Column number or column name of the explanatory variable
        obj (list): Column number or column name of the objective variable
        mid (list or None): Column number or column name of the intermediate variable
        scale (True or False): Whether to pipeline with StandardScaler
        algorithms (list): Specify learners and parameters for each learning. There are five levels of designation methods.
        1: 'GaussianProcessRegressor'
        2:  ['GaussianProcessRegressor',{'n_estimators':1}]
        3 : [['GaussianProcessRegressor'], ['GaussianProcessRegressor']]
        4 : [['GaussianProcessRegressor',{'n_estimators':1}], ['GaussianProcessRegressor', {}]]
        5 : [[['GaussianProcessRegressor',{'n_estimators':1}] for x in range(len(mid))], [['GaussianProcessRegressor',{'n_estimators7:1}] for x in range(len(Obj))]]
        1 is all the same algorithm
        2 are all the same parameters of the specified algorithm, and the parameters are specified with {}
        3 is [[exp → mid algorithm], [mid → obj algorithm]]
        4 includes parameter specification in addition to 3
        5 is all specified, [[[[algorithm to predict the first mid if exp → mid, {parameter}], [algorithm to predict the second mid of exp → mid , {Parameters}], ...,]
        [[Algorithm to predict the first of obj when mid → obj, {parameter}], [Algorithm to predict the second of obj when mid → obj, {parameter}], ...,] , ...,]
        If there is no mid,
        1 : 'GaussianProcessRegressor'
        2 : ['GaussianProcessRegressor', {}]
        3 : [[], ['GaussianProcessRegressor']]
        4 : [[], ['GaussianProcessRegressor', {}]]
        5 : [[], [['GaussianProcessRegressor',{'n_estimators':1}] for x in range(len(Obj))]]
        Or as in the case with mid. In this case, the parentheses on the back side are recognized as an algorithm of (exp → obj).        """
        self.exp = exp
        self.mid = mid
        self.obj = obj
        self.model_list = []  # List to store learned models
        
        # choose data
        data = pd.DataFrame(data)
        data_exp = self._choose_data(data, self.exp)
        data_obj = self._choose_data(data, self.obj)
        if self.mid is not None:
            data_mid = self._choose_data(data, self.mid)
        
        # The number of each variable
        len_exp = len(self.exp)
        if self.mid is None:
            len_mid = 0
        else:
            len_mid = len(self.mid)
        len_obj = len(self.obj)

        algorithms_ = self._algorithms_align_same_shape(len_mid, len_obj, algorithms)

        model_list1 = []  # models of exp→mid
        model_list2 = []  # models of mid→obj
        # make models of exp→mid
        if not (mid==None or mid==[] or mid==False):
            for i in range(len_mid):
                module =  algorithms_[0][i][0]
                param = algorithms_[0][i][1]
                model = eval(module + "(**param)")
                if scale:
                    steps = [('scale', StandardScaler()), ('est', model)]
                    algorithm = Pipeline(steps=steps)
                else:
                    algorithm = model

                algorithm.fit(data_exp, data_mid[:,i])
                model_list1.append(algorithm)

        # Create mid → obj model (if mid does not exist, exp → obj)
        for j in range(len_obj):
            module =  algorithms_[1][j][0]
            param = algorithms_[1][j][1]
            model = eval(module + "(**param)")
            if scale:
                steps = [('scale', StandardScaler()), ('est', model)]
                algorithm = Pipeline(steps=steps)
            else:
                algorithm = model

            if mid==None or mid==[] or mid==False:
                algorithm.fit(data_exp, data_obj[:,j])
            else:
                algorithm.fit(data_mid, data_obj[:,j])
            model_list2.append(algorithm)
        
        self.model_list = [model_list1, model_list2]
        return self.model_list