## Category variable and dummy variable and Undo ##

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

class CategoricalPreprocessing():
    
    def formatting(self, data, dummies=[], categories=[], category_order=[]):
        """
        data (numpy or DataFrame): Input data.
        categories (list[str or int] or None): Enter the column name or column number for categorical variable of nominal scale. Example : categories = ['carrier','substrate']
        category_order (dict or None): Enter the column name or column number for the categorical variable of the order scale.
        When specifying, {"column name": {"label": order}}。Example : category_order={'Sex':{'Male':0,'Female':1}}
        Make sure categories and category_order are not on the same column.
        dummies (list or None): Specify the column name or column number to be a dummy variable. Example : dummies=['material','carrier','substrate']
        The value converted by dummies comes to the front of the line.
        
        Return value
        data : Data after transformation.
        self.dummies_list : A list of the column numbers that get_dummies and what changed. [[Original column number, column name, [what changed]], ...]
        Example [[9, 'Race', ['White', 'Black']], [10, 'Sex', ['Male', 'Female']]]
        self.categories_reborn : Information necessary to restore the categorized variable.
        [[[Column name changed by category or column number], LabelEncorder defaultdict used for conversion], {category_order dictionary}]        例 [[[1, 2],defaultdict(sklearn.preprocessing.label.LabelEncoder,
              {'Education': LabelEncoder(), 'Martial_Status': LabelEncoder()})],
              {'Sex': {'Male': 0, 'Female': 1}}]
        self.categories : Column names of all columns that are categorized.
        
        dummies_list or categories_reborn is using in inverse_formatting
        dummies_list or category_colnames is using in make domain in GeneticAlgorithm
        """
        
        data = pd.DataFrame(data).copy()  # Convert to DataFrame for when input data is numpy
        
        # Convert according to categories
        le = LabelEncoder()
        if not(categories==None or categories==[] or categories==False):  # Whether to convert the nominal scale into categorical variables
            if all([type(x) == str for x in categories]):  # When category is column name
                le = defaultdict(LabelEncoder)
                data.loc[:,categories] = data.loc[:,categories].apply(lambda x: le[x.name].fit_transform(x))
                # Inverse the encoded
                #data.loc[:,categories].apply(lambda x: le[x.name].inverse_transform(x))
            elif all([type(x) == int for x in categories]):  # Whei category is column number
                le = defaultdict(LabelEncoder)
                data.iloc[:,categories] = data.iloc[:,categories].apply(lambda x: le[x.name].fit_transform(x))
            categories_reborn1 = [categories, le]  # List to undo
        else:
            categories = []
            categories_reborn1 = None

        category_column_list = []  # Save column name to categorize order scale
        if not (category_order==None or category_order==[] or category_order==False):  # Whether to categorize ordinal scale
            for column_name, value in category_order.items():  # Expand the input value dictionary into column names and order
                size_mapping = value
                if type(column_name)==str:
                    data.loc[:, column_name] = data.loc[:, column_name].map(size_mapping)
                elif type(column_name)==int:
                    data.iloc[:, column_name] = data.iloc[:, column_name].map(size_mapping)
                elif type(column_name)==float:
                    column_name = int(column_name)
                    data.iloc[:, column_name] = data.iloc[:, column_name].map(size_mapping)
                category_column_list.append(column_name)

        categories_ = categories +  category_column_list  # Save all converted column names in categories
        self.categories_reborn = [categories_reborn1, category_order]  # Save all information to return to categories_reborn
        
        if not(dummies==None or dummies==[] or dummies==False):  # When creating a dummy variable
            if all([type(x) == float for x in dummies]):
                dummies = [int(x) for x in dummies]
            
            dummies_column_list = []  # Save column names after dummy variable
            dummies_data_list = []  # Save DataFrame as dummy variable
            
            data_copy = data.copy()
            if all([type(x) == str for x in dummies]):
                for dummy in dummies:
                    colnum = data_copy.columns.get_loc(dummy)   # Column number of the place to be a dummy variable
                    colname = dummy   # Column name of the place to be a dummy variable
                    data_dummied = pd.get_dummies(data_copy.loc[:, dummy])  # to dummy variable
                    dummy_columns = list(data_dummied.columns.values)  # Get column name after dummy variable and store it in list
                    dummies_column_list.append([colnum, colname, dummy_columns])
                    dummies_data_list.append(data_dummied)
                    data.drop([dummy], axis = 1, inplace = True)  # Delete the specified column in the original data
            
            elif all([type(x) == int for x in dummies]):
                for dummy in dummies:
                    colnum = dummy  # Column number of the place to be a dummy variable
                    colname =  list(data_copy.iloc[:,[dummy]].columns)[0]  # Column name of the place to be a dummy variable
                    data_dummied = pd.get_dummies(data_copy.iloc[:, dummy])  # to dummy variable
                    dummy_columns = list(data_dummied.columns.values)  # Get column name after dummy variable and store it in list
                    dummies_column_list.append([colnum, colname, dummy_columns])
                    dummies_data_list.append(data_dummied)
                    data.drop([colname], axis = 1, inplace = True)  # Delete the specified column in the original data
        
        
            for dummies_data in dummies_data_list[::-1]:  # Iteratively process the dummy variable DataFrame to correspond to the column names
                data = pd.concat([dummies_data, data], axis = 1)  # Join so that dummy variable column comes first

        else:
            dummies_column_list = []  # If you do not want to make a dummy variable, output with [].
                                    
        self.dummies_list = dummies_column_list
        
        # It is difficult to use if the column name and column number are mixed, so convert it to the column name.
        category_colnames = []
        for category in categories_:
            if isinstance(category, int):
                column = list(data_copy.iloc[:,[category]].columns)[0]
                category_colnames.append(column)
            elif isinstance(category, str):
                category_colnames.append(category)
        self.category_colnames = category_colnames

        # Preformatted data, dummy columns and restoration information, categorization restoration information, categorized column names
        
        return data, self.dummies_list, self.categories_reborn, self.category_colnames
    
    
    def _reborn_dummies(self, dummies_data):
        """A function that returns the column name of the dummy variable df that is set to 1.
        That is, restoration of dummies. For apply."""
        dummies_data_index = list(dummies_data[dummies_data==1].index)
        if dummies_data_index==[]:
            return np.nan
        else:
            return dummies_data_index[0]
    
    def _pd_insert(self, original_df, insert_df, insert_index, axis):
        """Insert rows with pandas"""
        if axis==0:
            previous_df = original_df.iloc[:insert_index, :]
            behind_df = original_df.iloc[insert_index:,:]
            new_df = pd.concat([previous_df, insert_df, behind_df], axis=0)
        elif axis==1:
            previous_df = original_df.iloc[:, :insert_index]
            behind_df = original_df.iloc[:,insert_index:]
            new_df = pd.concat([previous_df, insert_df, behind_df], axis=1)
        return new_df
    
    def inverse_formatting(self, data, dummies_list='self', categories_reborn='self'):
        """
        Restore the processing performed in the preprocessing.
        data : (pandas) Data to be restored, processed with formatting.
        dummies_list : dummies_list generated by formatting
        categories_reborn : categories_reborn generated by formatting
        category_colnames has no turn. This is used when creating a domain with genetic_algorithm.
        """

        if dummies_list=='self' and hasattr(self, 'dummies_list'):
            # If it is defined in self, it is assigned as it is
            dummies_list = self.dummies_list
        elif dummies_list=='self' and not hasattr(self, 'dummies_list'):
            # If it is not defined, enter the default value for now.
            dummies_list = []
        if categories_reborn=='self' and hasattr(self, 'categories_reborn'):
            categories_reborn = self.categories_reborn
        elif categories_reborn=='self' and not hasattr(self, 'categories_reborn'):
            categories_reborn = [None, None]

        reborn_data = data.copy()

        # Restoration when dummy variables are used
        if not(dummies_list==[] or dummies_list==None or dummies_list==False):
            for i in range(len(dummies_list)):
                dummies_columns = dummies_list[i][2]  # Column name after conversion
                original_colname =  dummies_list[i][1]
                insert_colnum = dummies_list[i][0]
                dummies_data = data.loc[:,dummies_columns]

                reborn = pd.DataFrame(dummies_data.apply(self._reborn_dummies, axis=1), columns=[original_colname])

                reborn_data.drop(dummies_columns, axis=1, inplace=True)
                reborn_data = self._pd_insert(reborn_data, reborn, insert_colnum, axis=1)

        # Restoration of categorical variables converted with categories' LabelEncorder
        categories_reborn_le = categories_reborn[0]
        if not (categories_reborn_le==None or categories_reborn_le==[] \
                or categories_reborn_le==False):
            columns = categories_reborn_le[0]
            le = categories_reborn_le[1]

            if all([type(x) == int for x in columns]):
                # Inverse the encoded
                reborn_data.iloc[:, columns] = reborn_data.iloc[:, columns].astype(int)
                reborn_data.iloc[:, columns] = reborn_data.iloc[:, columns].apply(lambda x: le[x.name].inverse_transform(x))
            elif all([type(x) == str for x in columns]):
                # Inverse the encoded
                reborn_data.loc[:, columns] = reborn_data.loc[:, columns].astype(int)
                reborn_data.loc[:, columns] = reborn_data.loc[:, columns].apply(lambda x: le[x.name].inverse_transform(x))
            else:
                raise ValueError("Use same type or correct type")

        # Restoration of categorical variables converted by category_order dictionary
        categories_reborn_dict = categories_reborn[1]
        if not(categories_reborn_dict==None or categories_reborn_dict==[] \
                or categories_reborn_dict==False):
            # Originally, create a dictionary that reverses the conversion part of the conversion dictionary
            reborn_dict_swap = {}
            for key, items in categories_reborn_dict.items():
                d_swap = {v: k for k, v in items.items()}
                d_swap_ = {key: d_swap}
                reborn_dict_swap.update(d_swap_)

            # Convert as you would when converting
            for column_name, value in reborn_dict_swap.items():  # Expand the input value dictionary into column names and order
                size_mapping = value
                if type(column_name) == str:
                    reborn_data.loc[:, column_name] = reborn_data.loc[:, column_name].astype(int)
                    reborn_data.loc[:, column_name] = reborn_data.loc[:, column_name].map(size_mapping)
                elif type(column_name) == int:
                    reborn_data.iloc[:, column_name] = reborn_data.iloc[:, column_name].astype(int)
                    reborn_data.iloc[:, column_name] = reborn_data.iloc[:, column_name].map(size_mapping)
                elif type(column_name) == float:
                    column_name = int(column_name)
                    reborn_data.iloc[:, column_name] = reborn_data.iloc[:, column_name].astype(int)
                    reborn_data.iloc[:, column_name] = reborn_data.iloc[:, column_name].map(size_mapping)

        return reborn_data
    
    
    def change_columns_after_preprocess(self, exp, obj, mid=None, dummies_list='self'):
        """
        A function that converts the original exp, obj, and mid column numbers into column numbers after making them dummy variables, etc.
        exp : (list of int) original explanatory variable column number
        obj : (list of int) original objective variable column number
        mid : (list of int) original Intermediate variable column number
        It is not assumed that there will be dummy variables other than exp.
        Note the order of exp, obj, mid
        """
        if dummies_list=='self' and hasattr(self, 'dummies_list'):
            # If it is defined in self, it is assigned as it is
            dummies_list = self.dummies_list
        elif dummies_list=='self' and not hasattr(self, 'dummies_list'):
            # If it is not defined, enter the default value for now.
            dummies_list = []
        
        # If don't define
        if dummies_list==[] or dummies_list==None or dummies_list==False:
            return exp, obj, mid
        
        n_dummies = len(self.dummies_list)
        # Estimate extra columns with dummy variables
        x = 0
        for i in range(n_dummies):
            x += len(self.dummies_list[i][2])
        incre_num = x - n_dummies

        # shift exp and then others
        exp_shift = list(map(lambda x:x+incre_num, exp))
        exp_plus = [x for x in range(0, incre_num)]
        exp_after = exp_plus + exp_shift
        if mid==None or mid==[] or mid==False:
            mid_after = None
        else:
            mid_after = list(map(lambda x:x+incre_num, mid))
        obj_after = list(map(lambda x:x+incre_num, obj))
        
        return exp_after, obj_after, mid_after
    
    def str_for_int(self, original_df, exp, obj, mid=None):
        """
        Function to convert to int when the original exp, obj, and mid are column names.
        original_df : (pandas) original df
        exp : (list of str) Column name explanatory variable 
        obj : (list of str) Column name objective variable 
        mid : (list of str) Column name Intermediate variable 
        """
        exp_num = [original_df.columns.get_loc(x) if isinstance(x,str) else x for x in exp]
        obj_num = [original_df.columns.get_loc(x) if isinstance(x,str) else x for x in obj]
        if not(mid==None or mid==[] or mid==False):
            mid_num = [original_df.columns.get_loc(x) if isinstance(x,str) else x for x in mid]
        else:
            mid_num = None
        return exp_num, obj_num, mid_num
    