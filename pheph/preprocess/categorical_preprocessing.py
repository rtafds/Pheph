## Category variable and dummy variable and Undo ##

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

class CategoricalPreprocessing():
    
    def _sort_column_name_list(self, name_list, data_columns):
        """Sort column name list by column number.
        Args:
            name_list (list[str]): column name list
            data_columns (Index[]): data is pd.DataFrame, data_columns = data.columns 

        Returns:
            list[str]: sorted name list
        """
        columns_number_list = [data_columns.get_loc(x) for x in name_list]
        sorted_name_list = [name for _,name in sorted(zip(columns_number_list,name_list))]
        return sorted_name_list

    def _sort_nan_list_end(self, lst):
        """NAN is the end of the list containing NAN.
        Args:
            lst (list): One dimension list.
        """
        if len(lst)==0 or len(lst)==1 or not (np.nan in lst):
            return lst
        is_in_nan = False
        lst_sorted = []
        lst_sorted_append = lst_sorted.append
        for value in lst:
            if isinstance(value,str):
                lst_sorted_append(value)
            elif np.isnan(value):
                is_in_nan = True
            else:
                lst_sorted_append(value)
        if is_in_nan:
            lst_sorted_append(np.nan)
        return lst_sorted
    
    def _fill_not_defined_value(self, size_mapping, unique_value_set):
        """Automatically defines values that are not defined in data by Catetory_assign_dict.
        Args:
            size_mapping (dict): category_assign_dict_value
            unique_value_set (set): column unique set

        Returns:
            dict : adding not defined value dict.
        """
        
        size_mapping_key_set = set(size_mapping.keys())
        non_definition_values = list(unique_value_set - size_mapping_key_set)

        if len(non_definition_values)==0:
            # if fill all value, the process is end.
            return size_mapping

        size_mapping_max_value = max(size_mapping.values())
        non_definition_values_sorted = self._sort_nan_list_end(non_definition_values)

        # make dict, then add not defined value
        add_dict = {}
        for i,key in enumerate(non_definition_values_sorted):
            value = size_mapping_max_value + 1 + i
            add_dict[key] = value
        print(f"Add value {add_dict} to catogory_assign_dict")
        
        return {**size_mapping, **add_dict}
        
    def formatting(self, data, dummies=[], category_le_list=[], category_assign_dict={}):
        """
        data (numpy or DataFrame): Input data.
        category_le_list (list[str or int] or None): Enter the column name or column number for categorical variable of nominal scale. Example : category_le_list = ['carrier','substrate']
        category_assign_dict (dict or None): Enter the column name or column number for the categorical variable of the order scale.
          When specifying, {"column name": {"label": order}}。Example : category_assign_dict={'Sex':{'Male':0,'Female':1}}。
          Other Example : category_assign_dict = {2:{"0times":0,"1-2times":1,"3-5times":2,"6times_over":3,np.nan:4}, 3:{"0":0,">8000":1,">30000":2,"<=30000":3}}
          Make sure category_le_list and category_assign_dict are not on the same column.
        dummies (list or None): Specify the column name or column number to be a dummy variable. Example : dummies=['material','carrier','substrate']
        The value converted by dummies comes to the front of the line.
        
        Return value
        data : Data after transformation.
        self.dummies_list : A list of the column numbers that get_dummies and what changed. [[Original column number, column name, [what changed]], ...]
        Example [[9, 'Race', ['White', 'Black']], [10, 'Sex', ['Male', 'Female']]]
        self.categories_reborn : Information necessary to restore the categorized variable.
        [[[Column name changed by category or column number], LabelEncorder defaultdict used for conversion], {category_assign_dict dictionary}]        例 [[[1, 2],defaultdict(sklearn.preprocessing.label.LabelEncoder,
              {'Education': LabelEncoder(), 'Martial_Status': LabelEncoder()})],
              {'Sex': {'Male': 0, 'Female': 1}}]
        self.category_colnames : Column names of all columns that are categorized.
        
        dummies_list or categories_reborn is using in inverse_formatting
        dummies_list or category_colnames is using in make domain in GeneticAlgorithm
        """
        
        data = pd.DataFrame(data).copy()  # Convert to DataFrame for when input data is numpy
        data_columns = data.columns
        
        # Convert according to category_le_list
        le = LabelEncoder()
        if not(category_le_list==None or category_le_list==[] or category_le_list==False):  # Whether to convert the nominal scale into categorical variables
            if all([type(x) == str for x in category_le_list]):  # When category is column name
                category_le_list = self._sort_column_name_list(category_le_list, data_columns)
                le = defaultdict(LabelEncoder)
                data.loc[:,category_le_list] = data.loc[:,category_le_list].apply(lambda x: le[x.name].fit_transform(x))
                # Inverse the encoded
                #data.loc[:,category_le_list].apply(lambda x: le[x.name].inverse_transform(x))
            elif all([type(x) == int for x in category_le_list]):  # Whei category is column number
                category_le_list = sorted(category_le_list)
                le = defaultdict(LabelEncoder)
                data.iloc[:,category_le_list] = data.iloc[:,category_le_list].apply(lambda x: le[x.name].fit_transform(x))
            categories_reborn1 = [category_le_list, le]  # List to undo
        else:
            category_le_list = []
            categories_reborn1 = None

        category_column_list = []  # Save column name to categorize order scale
        if not (category_assign_dict==None or category_assign_dict=={} or category_assign_dict==False):  # Whether to categorize ordinal scale
            for column, value in category_assign_dict.items():  # Expand the input value dictionary into column names and order
                size_mapping = value
                if type(column)==str:
                    # If there is a value that is not defined, fill it automatically.
                    unique_value_set = set(data.loc[:, column].unique())
                    size_mapping = self._fill_not_defined_value(size_mapping, unique_value_set)
                    category_assign_dict[column] = size_mapping
                    data.loc[:, column] = data.loc[:, column].map(size_mapping)
                elif type(column)==int:
                    # If there is a value that is not defined, fill it automatically.
                    unique_value_set = set(data.iloc[:, column].unique())
                    size_mapping = self._fill_not_defined_value(size_mapping, unique_value_set)
                    category_assign_dict[column] = size_mapping
                    data.iloc[:, column] = data.iloc[:, column].map(size_mapping)
                elif type(column)==float:
                    column = int(column)
                    # If there is a value that is not defined, fill it automatically.
                    unique_value_set = set(data.iloc[:, column].unique())
                    size_mapping = self._fill_not_defined_value(size_mapping, unique_value_set)
                    category_assign_dict[column] = size_mapping  # 更新
                    data.iloc[:, column] = data.iloc[:, column].map(size_mapping)
                else:
                    raise ValueError("category_assign_dict keys are str or int")

        categories_ = category_le_list +  category_column_list  # Save all converted column names in category_le_list
        self.categories_reborn = [categories_reborn1, category_assign_dict]  # Save all information to return to categories_reborn
        
        if not(dummies==None or dummies==[] or dummies==False):  # When creating a dummy variable
            if all([type(x) == float for x in dummies]):
                dummies = [int(x) for x in dummies]
            
            dummies_column_list = []  # Save column names after dummy variable
            dummies_data_list = []  # Save DataFrame as dummy variable
            
            #data_copy = data.copy()
            if all([type(x) == str for x in dummies]):
                dummies = self._sort_column_name_list(dummies, data_columns)
                for dummy in dummies:
                    colnum = data_columns.get_loc(dummy)   # Column number of the place to be a dummy variable
                    colname = dummy   # Column name of the place to be a dummy variable
                    data_dummied = pd.get_dummies(data.loc[:, dummy])  # to dummy variable
                    dummy_columns = list(data_dummied.columns.values)  # Get column name after dummy variable and store it in list
                    dummies_column_list.append([colnum, colname, dummy_columns])
                    dummies_data_list.append(data_dummied)
                    data.drop([dummy], axis = 1, inplace = True)  # Delete the specified column in the original data
            
            elif all([type(x) == int for x in dummies]):
                dummies = sorted(dummies)
                for dummy in dummies:
                    colnum = dummy  # Column number of the place to be a dummy variable
                    colname =  list(data.iloc[:,[dummy]].columns)[0]  # Column name of the place to be a dummy variable
                    data_dummied = pd.get_dummies(data.iloc[:, dummy])  # to dummy variable
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
                column = list(data.iloc[:,[category]].columns)[0]
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
            insert_list = []
            len_dummies_list = len(dummies_list)
            for i in range(len_dummies_list):
                dummies_columns = dummies_list[i][2]  # Column name after conversion
                original_colname =  dummies_list[i][1]
                insert_colnum = dummies_list[i][0]
                dummies_data = data.loc[:,dummies_columns]

                reborn = pd.DataFrame(dummies_data.apply(self._reborn_dummies, axis=1), columns=[original_colname])

                reborn_data.drop(dummies_columns, axis=1, inplace=True)
                insert_list.append([insert_colnum, reborn])
            len_insert_list = len(insert_list)
            
            for i in range(len_insert_list):
                insert_colnum = insert_list[i][0]
                reborn = insert_list[i][1]
                reborn_data = self._pd_insert(reborn_data, reborn, insert_colnum, axis=1)

        columns_list = reborn_data.columns  # use column names for convert dtype is int
        # Restoration of categorical variables converted with category_le_list' LabelEncorder
        categories_reborn_le = categories_reborn[0]
        if not (categories_reborn_le==None or categories_reborn_le==[] \
                or categories_reborn_le==False):
            columns = categories_reborn_le[0]
            le = categories_reborn_le[1]

            if all([type(x) == int for x in columns]):
                # Inverse the encoded
        
                # convert dtype is int
                # For some reason the fllowing dosen't work reborn_data.iloc[:, columns] = reborn_data.iloc[:, columns].astype(int)
                columns_name = list(columns_list[columns])
                reborn_data[columns_name] = reborn_data[columns_name].astype("int64")
                reborn_data.iloc[:, columns] = reborn_data.iloc[:, columns].apply(lambda x: le[x.name].inverse_transform(x))
            elif all([type(x) == str for x in columns]):
                # Inverse the encoded
                reborn_data[columns] = reborn_data[columns].astype("int64")
                reborn_data.loc[:, columns] = reborn_data.loc[:, columns].apply(lambda x: le[x.name].inverse_transform(x))
            else:
                raise ValueError("Use same type or correct type")

        # Restoration of categorical variables converted by category_assign_dict dictionary
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
            for column, value in reborn_dict_swap.items():  # Expand the input value dictionary into column names and order
                size_mapping = value
                if type(column) == str:
                    reborn_data[:, column] = reborn_data[:, column].astype("int64")
                    reborn_data.loc[:, column] = reborn_data.loc[:, column].map(size_mapping)
                elif type(column) == int:
                    column_name = [columns_list[column]]
                    reborn_data[column_name] = reborn_data[column_name].astype("int64")
                    reborn_data.iloc[:, column] = reborn_data.iloc[:, column].map(size_mapping)
                elif type(column) == float:
                    column = int(column)
                    column_name = [columns_list[column]]
                    reborn_data[column_name] = reborn_data[column_name].astype("int64")
                    reborn_data.iloc[:, column] = reborn_data.iloc[:, column].map(size_mapping)

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
    