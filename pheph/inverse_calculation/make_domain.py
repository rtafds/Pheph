import random
import numpy as np
import pandas as pd

class MakeDomain():

    def _domain_from_dummies_and_categories(self, data, exp, dummies_list, category_colnames):
        """
        Make domain from dummies_list and category_colnames made by formatting function in Preprocess class.
        dummies_list : (list) dummy variable original column number and after column name. Made by formatting function in Preprocess class.
        category_colnames : (list) category variable column names. Made by formatting function in Preprocess class. 
        """
        # max exp column number
        max_exp_colnum = np.array(exp).max()
        exp_colnames = list(data.iloc[:,exp].columns)
        
        # Process dummy variables
        domain = {}
        x = 0  # Since the dummy variable is specified at the top, search for the dummy variable column with x.
        # make the list as fallows : ('choice2', (0,1), (1,0)) 
        len_dummies = len(dummies_list)
        for i in range(len_dummies):
            for j in range(2,100):
                len_dummies_list_i = len(dummies_list[i][2])
                if len_dummies_list_i==j:
                    dummy_domain = []
                    dummy_domain_append = dummy_domain.append
                    for k in range(len_dummies_list_i):
                        du_dom = [0 for x in range(len_dummies_list_i)]
                        du_dom[len_dummies_list_i - k - 1] = 1
                        dummy_domain_append(tuple(du_dom))
                    dummy_domain = tuple(dummy_domain)

                    domain[x] = ('choice2', dummy_domain)
                    x += j
                    # if domain num is out of range of exp, domain isn't made.
                    if x > max_exp_colnum:
                        break

        # Processing categorical variables
        # For categorical variables, search for the column number changed by the dummy variable.
        incre_dummy = x - len(dummies_list)   # Number of columns increased by dummy variable
        dummies_num = [z[0] for z in dummies_list]  # Original column number of the dummy variable

        categories_col = []  # Save categorical variable locations (to avoid later)
        for col in category_colnames:
            # if category_colnames if out of range of exp, domain isn't made.
            if not col in exp_colnames:
                continue
            # When the column number is used, the amount of the dummy variable 
            # that comes out at the top and the amount that the dummy variable increases are considered.
            if isinstance(col, int):
                missed_dummy_num = sum([1 for y in dummies_num if y>col])  # Number of columns affected by missing dummy variable
                incre_num = incre_dummy + missed_dummy_num  # How much do you have to shift the line
                col_ = col + incre_num
                df_col = data.iloc[:, col_]
                domain[col_] = ('randint',(min(df_col), max(df_col)))
                categories_col.append(col_)
            # If column names, don't worry
            elif isinstance(col, str):
                df_col = data.loc[:, col]
                col_num = data.columns.get_loc(col)
                domain[col_num] = ('randint',(min(df_col), max(df_col)))
                categories_col.append(col_num)
                
        return domain, x, categories_col, incre_dummy, dummies_num
    
    def _domain_not_defined_part(self, domain, data, exp, x, categories_col, domains_col=[], extrapolation=0.1):
        """
        Make domain from dummies_list and category_colnames made by formatting function in Preprocess class.
        dummies_list : (list) dummy variable original column number and after column name. Made by formatting function in Preprocess class.
        category_colnames : (list) category variable column names. Made by formatting function in Preprocess class. 
        """
        # The other parts are made of uniform.
        for col in exp:
            if col < x:
                continue
            elif col in categories_col:
                continue
            elif col in domains_col:
                continue
            df_col = data.iloc[:, col]
            width = max(df_col) - min(df_col)
            if min(df_col)==0.0:
                min_ = 0.0
                max_ = max(df_col)+width*extrapolation*2
            else:
                min_ = min(df_col)-width*extrapolation
                max_ = max(df_col)+width*extrapolation
            domain[col] = ('uniform',(min_, max_))
        domain = dict(sorted(domain.items(), key=lambda x:x[0]))
        return domain
        
    def make_domain_auto(self, data, exp, dummies_list='self', category_colnames='self', extrapolation=0.1):
        """A function that automatically acquires the domain.
        If there is no self.dummies_list and self.category_colnames, it is treated as not.
        If dummy variables and categorical variables are preprocessed and registered in the class, the information is used.
        Go to create random.uniform with maximum and minimum values for undefined places. At that time, an extrapolated area multiplied by extrapolation is defined as a definition area.
        Example: When the minimum value is 20 and the maximum value is 120, extrapolation = 0.1 extrapolates 100 * 0.1 minutes, and 10 ~ 130 is the domain.
        data : (pandas) data after pretreatment
        exp : (list) List of column numbers for experimental conditions
        Dependence ::: Depends on dummies_list and category_colnames created by preprocessing. If not, treat it as missing.
        """
        data = pd.DataFrame(data)
            
        if dummies_list=='self' and hasattr(self, 'dummies_list'):
            # If it is defined in self, it is assigned as it is
            dummies_list = self.dummies_list
        elif dummies_list=='self' and not hasattr(self, 'dummies_list'):
            # If it is not defined, enter the default value for now.
            dummies_list = []
            
        if category_colnames=='self' and hasattr(self, 'category_colnames'):
            category_colnames = self.category_colnames
        elif category_colnames=='self' and not hasattr(self, 'category_colnames'):
            category_colnames = []
        
        domain, x, categories_col, incre_dummy, dummies_num = self._domain_from_dummies_and_categories(data, exp, dummies_list, category_colnames)
        domain = self._domain_not_defined_part(domain, data, exp, x, categories_col, [], extrapolation=extrapolation)
            
        return domain
    
    def domain_str_for_int(self, original_data, domain_original):
        """Change column name of domain_original to column number because domain does not support other than column numbers.
        original_df : (pandas) original df. those that are not dummy variables and categorical variables.
        """
        domain_original_int = {}
        for col in domain_original.keys():
            if isinstance(col, int):
                domain_original_int[col] = domain_original[col]
            elif isinstance(col, str):
                col_num = original_data.columns.get_loc(col)
                domain_original_int[col_num] = domain_original[col]
        domain_original_int = dict(sorted(domain_original_int.items(), key=lambda x:x[0]))
        return domain_original_int
    
    def make_domain_from_original(self, data, exp, domain_original, 
                                   dummies_list='self', category_colnames='self', extrapolation=0.1, original_data=[]):
        """Create domain from domain_original.
        Depends on dummies_list and category_colnames created by preprocessing. If not, treat it as missing.
        domain_original is the domain created for the original data.
        A domain is created for a data column before making it a dummy variable or categorical variable.
        As specifications, dummy variables and categorical variables are automatically created by adding dummy variables and categorical variables, 
        and adding or removing dummy variables. Create the missing part in the same way as the make_domain_auto function.
        
        The domain is described by the following dictionary. For the original exp = [x for x in range (0,11)], exp = [x for x in range (0,16)] after the dummy category,
        domain_original = {
              3:('uniform', (0.5, 0.6)),
              5:('randrange',(80,180,10)),
              6:('randint',(1150, 1180)),
              7:('choice', tuple([0]+[x for x in range(850, 1000, 10)])),
              8:('choice', tuple([0] + [x for x in range(200, 350, 10)])),
              9:('choice', tuple([0]+ [x for x in range(500, 700, 10)])),
             10:('uniform', (1600, 10000)) }
        On the other hand, if the original [0,1,3] is made a dummy variable and [2,4] is made a categorical variable
        {0: ('choice2', ((0, 1), (1, 0))),
             2: ('choice2', ((0, 1), (1, 0))),
             4: ('choice2', ((0, 0, 1), (0, 1, 0), (1, 0, 0))),
             7: ('randint', (0, 2)),
             8: ('randint', (0, 4)),
             9: ('randrange', (80, 180, 10)),
             10: ('randint', (1150, 1180)),
             11: ('choice', tuple([0]+[x for x in range(850, 1000, 10)])),
             12: ('choice', tuple([0] + [x for x in range(200, 350, 10)])),
             13: ('choice', tuple([0]+ [x for x in range(500, 700, 10)])),
             14: ('uniform', (1600, 10000)),
             15: ('uniform', (0.0, 0.0625)) }
             (15 is the complement from the domain) (3 is batted with the categorical variable, so the categorical variable has priority)
             In this way, categorical variable + dummy variable + shift to create the original + completion, and complete the domain.
        choice is random choice in 1d tuple
        choice2 is random choice in 2d tuple
        """
        data = pd.DataFrame(data)
        # If dummies_list is not defined
        if dummies_list=='self' and hasattr(self, 'dummies_list'):
            # If it is defined in self, it is assigned as it is
            dummies_list = self.dummies_list
        elif dummies_list=='self' and not hasattr(self, 'dummies_list'):
            # If it is not defined, enter the default value for now.
            dummies_list = []
            
        if category_colnames=='self' and hasattr(self, 'category_colnames'):
            category_colnames = self.category_colnames
        elif category_colnames=='self' and not hasattr(self, 'category_colnames'):
            category_colnames = []
            
        #  Change column name of domain_original to column number
        if isinstance(original_data, pd.DataFrame):
            domain_original = self.domain_str_for_int(original_data, domain_original)
        
        domain, x, categories_col, incre_dummy, dummies_num = self._domain_from_dummies_and_categories(data, exp, dummies_list, category_colnames)

        # sort by column number
        domain_original = dict(sorted(domain_original.items(), key=lambda x:x[0]))

        # Add the domain_original minutes.
        domains_col = []
        for j in domain_original.keys():
            missed_dummy_num = sum([1 for y in dummies_num if y>j])  # Number of columns affected by missing dummy variable
            incre_num = incre_dummy + missed_dummy_num  # How much do you have to shift the line.
            # If dummy variables and categorical variables are specified, give priority to them
            if j+incre_num < x:
                continue
            elif j+incre_num in categories_col:
                continue
            domain[j+incre_num] = domain_original[j]
            domains_col.append(j+incre_num)

        domain = dict(sorted(domain.items(), key=lambda x:x[0]))

        domain = self._domain_not_defined_part(domain, data, exp, x, categories_col, domains_col, extrapolation=extrapolation)

        return domain
    