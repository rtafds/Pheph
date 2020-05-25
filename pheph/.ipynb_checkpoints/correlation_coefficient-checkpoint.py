## calculating correlationcoefficient ##
# using sitiuation is Transfer learning.
# Can be used to fill missing values.

import numpy as np
import pandas as pd
from minepy import MINE
from pyHSICLasso import HSICLasso
from sklearn.utils import check_array


class CorrelationCoefficient():
    
    def mic_matrix(self, data, n_sample=False, frac_sample=False):
        '''Get mic correlation coefficient matrix
        Calculated coefficients are saved as instances.
        data: (numpy or pandas) A data frame that contains all explanatory and objective variables
        n_sample : (int) How much random sampling to do. False if not.
        If a numerical value is entered, sampling is performed using that number of rows.
        frac_sample: [0 ~ 1] (float) Sampled as a percentage of the number of rows. Not used at the same time as n_sample.
        '''
        data = pd.DataFrame(data).copy()
        data = data.dropna()  # Delete missing values and think
        # Sampling when n_sample contains a numerical value
        if not n_sample:
            if not frac_sample:
                # n_sample=False, frac_sample=False
                pass
            else:
                # n_sample=False, frac_sample=int
                data = data.sample(frac=frac_sample, replace=True)
        else:    
            
            if not frac_sample:
                # n_sample=int, frac_sample=False
                data = data.sample(n=n_sample, replace=True)
            else:
                # n_sample=int, frac_sample=int
                raise ValueError('Please enter a value for `frac` OR `n`, not both')
        
        data = check_array(data, accept_sparse="csc", dtype=float)  #  numpy.ndarrayに変換
        n_col = data.shape[1]
        mic_array = []
        mic_append = mic_array.append
        for i in range(n_col):
            temp_mic = []  # list to store one line of mic
            temp_mic_append = temp_mic.append
            for j in range(n_col):
                # Calculate only one of them because it is symmetric
                if i>=j:
                    temp_mic_append(0.0)
                else:
                    mine=MINE()
                    mine.compute_score(data[:, i], data[:, j])
                    temp_mic_append(mine.mic())
            mic_append(temp_mic)
            
        mic_ = np.array(mic_array)
        # Create a correlation coefficient matrix by copying the symmetry of the upper triangular matrix → transpose → unit matrix.
        self.mic = mic_ + mic_.T + np.eye(N=n_col, dtype=float)
        return self.mic
    
    def mine_matrix(self, data ,n_sample=False, frac_sample=False):
        '''Obtains all coefficient values related to mine as a correlation coefficient matrix.
        Calculated coefficients are saved as instances.
        data: (numpy or pandas) A data frame that contains all explanatory and objective variables
        n_sample : (int) How much random sampling to do. False if not.
        If a numerical value is entered, sampling is performed using that number of rows.
        frac_sample: [0 ~ 1] (float) Sampled as a percentage of the number of rows. Not used at the same time as n_sample.
        '''
        data = pd.DataFrame(data).copy()
        data = data.dropna()  # Delete missing values and think
        # Sampling when n_sample contains a numerical value
        if not n_sample:
            if not frac_sample:
                # n_sample=False, frac_sample=False
                pass
            else:
                # n_sample=False, frac_sample=int
                data = data.sample(frac=frac_sample, replace=True)
        else:    
            
            if not frac_sample:
                # n_sample=int, frac_sample=False
                data = data.sample(n=n_sample, replace=True)
            else:
                # n_sample=int, frac_sample=int
                raise ValueError('Please enter a value for `frac` OR `n`, not both')

        data = check_array(data, accept_sparse="csc", dtype=float)  # Convert to numpy.ndarray
        n_col = data.shape[1]
        mic_array = []  # Nonlinear correlation
        mas_array = []  # Linearity
        mev_array = []  # Functionality
        mic_append = mic_array.append  # Put append outside the loop and it will be a little faster
        mas_append = mas_array.append
        mev_append = mev_array.append
        
        for i in range(n_col):
            temp_mic = []
            temp_mas = []
            temp_mev = []
            
            temp_mic_append = temp_mic.append
            temp_mas_append = temp_mas.append
            temp_mev_append = temp_mev.append
            
            for j in range(n_col):
                if i>=j:
                    temp_mic_append(1.0)
                    temp_mas_append(1.0)
                    temp_mev_append(1.0)
                else:
                    mine=MINE()
                    mine.compute_score(data[:, i], data[:, j])
                    temp_mic_append(mine.mic())
                    temp_mas_append(mine.mas())
                    temp_mev_append(mine.mev())
        
            mic_append(temp_mic)
            mas_append(temp_mas)
            mev_append(temp_mev)
        mic_ = np.array(mic_array)
        mas_ = np.array(mas_array)
        mev_ = np.array(mev_array)
        self.mic = mic_ + mic_.T + np.eye(N=n_col, dtype=float)
        self.mas = mas_ + mas_.T + np.eye(N=n_col, dtype=float)
        self.mev = mev_ + mev_.T + np.eye(N=n_col, dtype=float)
        
        self.corr = np.corrcoef(data,rowvar=False)  # Pearson's correlation coefficient
        self.corr[np.isnan(self.corr)] = 1  # If the data values are all the same, NaN is used, so fill with 1 appropriately.
        self.mic_r2 = self.mic - self.corr  # Degree of nonlinearity
        return self.mic, self.mas, self.mev, self.mic_r2, self.corr
        
    def hsic_lasso_matric(self, data, n_jobs=2, n_sample=False, frac_sample=False):
            
        '''Calculate hsic lasso (subtract correlation between explanatory variables).
        Since the correlation coefficient matrix is not symmetric, it is viewed in the row direction.
        The correlation between variable 0 and the other variable is stored as the component on the 0th row,
        and the correlation between variable 1 and the other variable is stored as the component on the first row.
        
        n_jobs : (int) Indicates the number of cores to be calculated. -1 for GPU.
        data: (numpy or pandas) A data frame that contains all explanatory and objective variables
        n_sample : (int) How much random sampling to do. False if not.
        If a numerical value is entered, sampling is performed using that number of rows.
        frac_sample: [0 ~ 1] (float) Sampled as a percentage of the number of rows. Not used at the same time as n_sample.
        '''
        data = copy(data)
        data = pd.DataFrame(data).dropna()
        # Sampling when n_sample contains a numerical value
        if not n_sample:
            if not frac_sample:
                # n_sample=False, frac_sample=False
                pass
            else:
                # n_sample=False, frac_sample=int
                data = data.sample(frac=frac_sample, replace=True)
        else:    
            
            if not frac_sample:
                # n_sample=int, frac_sample=False
                data = data.sample(n=n_sample, replace=True)
            else:
                # n_sample=int, frac_sample=int
                raise ValueError('Please enter a value for `frac` OR `n`, not both')

        data = check_array(data, accept_sparse="csc", dtype=float)  # Convert to numpy.ndarray
        n_col = data.shape[1]
        hsic_array = np.empty((0, n_col-1), float)
        for i in range(n_col):
            X = np.delete(data, obj=i, axis=1)
            y = data[:, i]
            
            # Calculation of hsic_lasso
            hsic_lasso = HSICLasso()
            hsic_lasso.input(X, y)
            hsic_lasso.regression(num_feat=X.shape[1], discrete_x=False, n_jobs=n_jobs)
            # hsic_lasso only appears in descending order of score, so sort
            hsic_ = np.array([hsic_lasso.get_index(), hsic_lasso.get_index_score()])
            hsic_ = hsic_.T  # Transpose because it is difficult to use
            # Since there are not enough scores that came out, add 0.0 to the index to complement
            lack_set = set([x for x in range(X.shape[1])]) - set(hsic_[:,0])
            for lack in lack_set:
                lack_list = np.array([[lack, 0.0]])
                hsic_ = np.append(hsic_, lack_list, axis=0)
            hsic_ = hsic_[np.argsort(hsic_[:,0])]  # Sort by index
            hsic_array = np.append(hsic_array, hsic_[:,1].reshape(1,-1), axis=0)
        # Since it does not include the correlation component with itself, add 1.0
        n_row = hsic_array.shape[0]
        for i in range(n_row):
            insert_i = (n_row+1)*i
            hsic_array = np.insert(hsic_array, insert_i, 1.0)
        self.hsic_lasso = hsic_array.reshape(n_row,-1)
        return self.hsic_lasso