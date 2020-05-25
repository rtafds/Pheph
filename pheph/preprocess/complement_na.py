import numpy as np
import pandas as pd
# If you import here, you can use it.
from sklearn.linear_model import LogisticRegression, HuberRegressor, LinearRegression,Ridge,Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor


class ComplementNa():

    def _fillna_X_and_make_X_predict(self, data, col):
        """Supplement explanatory variables with missing values once with the average value of the column.
        data : (pandas) input data
        col : columns having NaN, and this column row predict other columns."""
        data_columns = data.columns
        y = data.loc[:,[col]]
        X = data.drop(col, axis=1)
        X.fillna(X.mean(), inplace=True)
        data_fillna_X = pd.concat([X,y], axis=1)
        data_fillna_X_for_training = data_fillna_X.loc[:, data_columns]  # Sort in original column order
        y_ = y.iloc[:,0]  # reshape
        # Extract rows with missing values in the objective variable from the newly complemented data
        row_y_is_na = data_fillna_X[y_.isnull()]
        X_fillna_for_predict = row_y_is_na.drop(col, axis=1)
        return data_fillna_X_for_training, X_fillna_for_predict

    def _X_y_split(self, data, col):
        """
        Divide training data with missing values into explanatory variables and objective variables.
        data : (pandas) input data
        col: columns having NaN, and this column row predict other columns.
        """
        y = data.loc[:, col]  # The data in the specified column having NaN
        data_dropna_having_y = data[~y.isnull()]  # Data with deleted rows with missing values in specified column
        X_train = data_dropna_having_y.drop(col, axis = 1)  # Data excluding specified columns (candidate variables for training data)
        y_train = data_dropna_having_y.loc[:, col]  # Specified column (objective variable of training data)
        return X_train, y_train.values  # Training data

    def _X_y_split_dropna_all(self, data, col):
        """
        Of the original data that is not supplemented with missing values, 
        data with no missing values is divided into explanatory variables and objective variables as training data.
        data : (pandas)input data
        i : columns having NaN, and this column row predict other columns.
        """
        data_dropna_all = data[~data.isnull().any(axis=1)]  # Data with no missing values
        X_train = data_dropna_all.drop(i, axis = 1)  # Data excluding specified columns (candidate variables for training data)
        y_train = data_dropna_all.iloc[:, i]  # Specified column (objective variable of training data)
        return X_train, y_train.values  # Training data
    
    def _algorithm_to_same_shape(self, algorithm):
        """
        Regardless of what type of algorithm comes in, change to the pattern 4.
        1: 'RandomForest'
        2:  ['RandomForest',{'n_estimators':1}]
        3 : [['RandomForestRegressor'], ['RandomForestClassifier']]
        4 : [['RandomForestRegressor',{'n_estimators':1}], ['RandomForestClassifier', {}]]
        algorithm : (list or str) object of pattern 1~4 
        """
        # Judgment based on the type of np.array
        if np.array(algorithm).shape==() and np.array(algorithm[1]).shape==():
            # For pattern 1
            # for making mistake 
            if 'Regressor' in algorithm:
                algorithm = algorithm.replace('Regressor','')
            elif 'Classifier' in algorithm:
                algorithm = algorithm.replace('Classifier','')
            algorithm_ = [[algorithm+'Regressor',{}],[algorithm+'Classifier',{}]]
        elif np.array(algorithm).shape==(2,) and np.array(algorithm[1]).shape==():
            # For pattern 2
            if 'Regressor' in algorithm[0]:
                algorithm[0] = algorithm[0].replace('Regressor','')
            elif 'Classifier' in algorithm[0]:
                algorithm[0] = algorithm[0].replace('Classifier','')
            algorithm_ = [[algorithm[0]+'Regressor',algorithm[1]],[algorithm[0]+'Classifier',algorithm[1]]]
        elif np.array(algorithm).shape==(2,1) and np.array(algorithm[1]).shape==(1,):
            # For pattern 3
            # for making mistake to reverse
            if 'Regressor' in algorithm[1] and 'Classifier' in algorithm[0]:
                copy_algorithm = algorithm.copy()
                algorithm[0] = copy_algorithm[1]
                algorithm[1] = copy_algorithm[0]
            algorithm_ = [[algorithm[0], {}], algorithm[1],{}]
        elif np.array(algorithm).shape==(2,2) and np.array(algorithm[1]).shape==(2,):
            # For pattern 4
            # for making mistake to reverse
            if 'Regressor' in algorithm[1][0] and 'Classifier' in algorithm[0][0]:
                copy_algorithm = algorithm.copy()
                algorithm[0] = copy_algorithm[1]
                algorithm[1] = copy_algorithm[0]
            algorithm_ = algorithm
        else:
            raise ValueError("algorithm shape is incorrect")

        return algorithm_
    
    def _predict_na(self, X_train, y_train, X_predict, col, category_colnames, algorithm, scale=True):
        """
        Data in nan columns is predicted other rows.
        if i columns data is category variable, using Classification. 
        It is assumed that dummies variable is dropped all by preprocessing (if dummy variable is NaN, all variable dummied is 0).
        X_train : explanation variable for training.
        y_train : objective variable (column having nan).
        X_predict : nan data raws
        col : nan data (y) column number
        category_colnames : All categorical variable number.
        algorithm : (list) using algorithm shaped by _algorithm_to_same_shape.
        scale : (bool) Whether StandardScaler is using.
        """
        # Classification if there is an explanatory variable in category_colnames_number, otherwise regression
        if col in category_colnames: 
            module =  algorithm[1][0]
            param = algorithm[1][1]
            model = eval(module + "(**param)")
            if scale:
                steps = [('scale', StandardScaler()), ('est', model)]
                model_c = Pipeline(steps=steps)
            else:
                model_c = model
            model_c.fit(X_train, y_train)
            predict_nan = model_c.predict(X_predict)
        else:  # regression
            module =  algorithm[0][0]
            param = algorithm[0][1]
            model = eval(module + "(**param)")
            if scale:
                steps = [('scale', StandardScaler()), ('est', model)]
                model_r = Pipeline(steps=steps)
            else:
                model_r = model
            model_r.fit(X_train, y_train)
            predict_nan = model_r.predict(X_predict)
        return predict_nan
        
    def complena(self, data, corr=None, category_colnames= 'self', algorithm=['RandomForest',{'n_estimators':100}], scale=True, 
                   decision_interpolation = True):
        """
        data : (numpy or pandas) Input data
        corr : (numpy or pandas) Correlation coefficient of input data
        category_colnames: (list) Specify column name of categorical variable.
        algorithm : (list or str) specified using ML algorithm as follows pattern. Specify regressor and classifier.
        1: 'RandomForest'
        2:  ['RandomForest',{'n_estimators':100}]
        3 : [['RandomForestRegressor'], ['RandomForestClassifier']]
        4 : [['RandomForestRegressor',{'n_estimators':10}], ['RandomForestClassifier', {}]]
        scale : (bool) True : pipeline of StandardScaler 
        decision_interpolation : (True) Complement all missing values of explanatory variables and use them for all training data
                                (False) Do not use data with missing values for explanatory variables, use only training data with no remaining missing values
        """
        if category_colnames=='self' and hasattr(self, 'category_colnames'):
            category_colnames= self.category_colnames
        elif category_colnames=='self' and not hasattr(self, 'category_colnames'):
            category_colnames= []

        # algorithm to same shape
        algorithm_ = self._algorithm_to_same_shape(algorithm)

        data = pd.DataFrame(data).copy()
        data_colnames = data.columns  # Original data column order

        # Missing values and numeric column names for each column
        n_nan = data.isnull().sum(axis = 0)

        # Sort by most missing values
        number_of_nan = pd.DataFrame({"n_nan": n_nan}).T
        plus_n_nan_data = pd.concat([data, number_of_nan]).sort_values(by = "n_nan", axis = 1, ascending = False)
        sorted_colnames = plus_n_nan_data.columns
        sorted_n_nan = plus_n_nan_data.loc["n_nan", :]

        for col, value in zip(sorted_colnames,sorted_n_nan):
            if value > 0:  # If there are missing values
                if decision_interpolation == True:  # Complement all missing values of explanatory variables and use them for all training data
                    data_fillna_X_for_training, X_fillna_for_predict = self._fillna_X_and_make_X_predict(data, col)

                    X_train, y_train = self._X_y_split(data_fillna_X_for_training, col)

                    predict_nan = self._predict_na(X_train, y_train, X_fillna_for_predict, \
                                                             col, category_colnames, algorithm=algorithm_, scale=scale)

                # Do not use data with missing values for explanatory variables, use only training data with no remaining missing values
                elif decision_interpolation == False:
                    _, X_fillna_for_predict = self._fillna_X_and_make_X_predict(data, col)

                    X_train, y_train = self._X_y_split_dropna_all(data, col)

                    predict_nan = self._predict_na(X_train.values, y_train, X_fillna_for_predict, col, category_colnames, \
                                                            algorithm=algorithm_, scale=scale)

                replace_col = data.loc[:, col]  # Extract columns with missing values
                null_place = replace_col.isnull()  # Bool with missing value as True
                data.loc[:, col][null_place] = predict_nan

        data = data.reset_index(drop=True)

        return data
