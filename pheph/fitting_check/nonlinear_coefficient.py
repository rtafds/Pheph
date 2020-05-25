## calculating correlationcoefficient ##
# using sitiuation is Transfer learning.
# Can be used to fill missing values.

# using https://github.com/amber0309/HSIC/blob/master/HSIC.py
# MIT License

#Copyright (c) 2016 Shoubo

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
import pandas as pd
from scipy.stats import gamma
from minepy import MINE

def mine_matrix(data , mode='all', n_sample=False, frac_sample=False):
    """Obtains all coefficient values related to mine as a correlation coefficient array.
    Calculated coefficients are saved as instances.
    mode : (str or list of str) Specify what to calculate, 'mic', 'mas', 'mev', 'corr', 'mic_r2'
    if mode='all', calculation all.
    data: (numpy or pandas) A data frame that contains all explanatory and objective variables
    n_sample : (int) How much random sampling to do. False if not.
    If a numerical value is entered, sampling is performed using that number of rows.
    frac_sample: [0 ~ 1] (float) Sampled as a percentage of the number of rows. Not used at the same time as n_sample.
    """
    if mode=='all':
        mode=['mic', 'mas', 'mev', 'corr', 'mic_r2']
    elif isinstance(mode, str):
        mode = [mode]

    data = np.array(data)
    data = data[~np.isnan(data).any(axis=1),:]  # Delete rows contain missing values
    # Sampling when n_sample contains a numerical value
    # Both definitions
    if n_sample and frac_sample:
        raise ValueError('n_sample and frac_sample don`t using both')
    elif not n_sample and frac_sample:
        # n_sample=False, frac_sample=int
        data = data.sample(frac=frac_sample, replace=True)
    elif n_sample and not frac_sample:
        # n_sample=int, frac_sample=False
        data = data.sample(n=n_sample, replace=True)
    # else is pass

    n_col = data.shape[1]
    mic = []  # Nonlinear correlation
    mic_append = mic.append  # Put append outside the loop and it will be a little faster
    mas = []  # Linearity
    mas_append = mas.append
    mev = []  # Functionality
    mev_append = mev.append

    for i in range(n_col):
        
        mic_row = []
        mic_row_append = mic_row.append
        mas_row = []
        mas_row_append = mas_row.append
        mev_row = []
        mev_row_append = mev_row.append

        for j in range(n_col):
            if i>=j:
                mic_row_append(1.0)
                mas_row_append(1.0)
                mev_row_append(1.0)
            else:
                mine=MINE()
                mine.compute_score(data[:, i], data[:, j])
                mic_row_append(mine.mic())
                mas_row_append(mine.mas())
                mev_row_append(mine.mev())

        mic_append(mic_row)
        mas_append(mas_row)
        mev_append(mev_row)
    
    returns = []
    if 'mic' in mode or 'mic_r2' in mode:
        mic_ = np.array(mic)
        mic_array = mic_ + mic_.T + np.eye(N=n_col, dtype=float)
        if 'mic' in mode:
            returns.append(mic_array)
    if 'mas' in mode:
        mas_ = np.array(mas)
        mas_array = mas_ + mas_.T + np.eye(N=n_col, dtype=float)
        returns.append(mas_array)
    if 'mev' in mode:
        mev_ = np.array(mev)
        mev_array = mev_ + mev_.T + np.eye(N=n_col, dtype=float)
        returns.append(mev_array)
    if 'corr' in mode or 'mic_r2' in mode:
        corr_array = np.corrcoef(data,rowvar=False)  # Pearson's correlation coefficient
        corr_array[np.isnan(corr_array)] = 1  # If the data values are all the same, NaN is used, so fill with 1 appropriately.
        if 'corr' in mode:
            returns.append(corr_array)
    if 'mic_r2' in mode:
        mic_r2_array = mic_array - corr_array  # Degree of nonlinearity
        returns.append(mic_r2_array)
    return returns


def rbf_dot(pattern1, pattern2, deg):
    """fof hsic_gam"""
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = np.sum(pattern1*pattern1, 1).reshape(size1[0],1)
    H = np.sum(pattern2*pattern2, 1).reshape(size2[0],1)
    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))

    H = Q + R - 2* np.dot(pattern1, pattern2.T)
    H = np.exp(-H/2/(deg**2))

    return H


def hsic_gam(X, Y, alph = 0.5):
    """
    X, Y are numpy vectors with row - sample, col - dim
    alph is the significance level
    auto choose median to be the kernel width
    """
    n = X.shape[0]

    # ----- width of X -----
    Xmed = X

    G = np.sum(Xmed*Xmed, 1).reshape(n,1)
    Q = np.tile(G, (1, n) )
    R = np.tile(G.T, (n, 1) )

    dists = Q + R - 2* np.dot(Xmed, Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_x = np.sqrt( 0.5 * np.median(dists[dists>0]) )
    # ----- -----
    # ----- width of X -----
    Ymed = Y

    G = np.sum(Ymed*Ymed, 1).reshape(n,1)
    Q = np.tile(G, (1, n) )
    R = np.tile(G.T, (n, 1) )

    dists = Q + R - 2* np.dot(Ymed, Ymed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_y = np.sqrt( 0.5 * np.median(dists[dists>0]) )
    # ----- -----

    bone = np.ones((n, 1), dtype = float)
    H = np.identity(n) - np.ones((n,n), dtype = float) / n

    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)
    Kc = np.dot(np.dot(H, K), H)
    Lc = np.dot(np.dot(H, L), H)

    testStat = np.sum(Kc.T * Lc) / n

    varHSIC = (Kc * Lc / 6)**2
    varHSIC = ( np.sum(varHSIC) - np.trace(varHSIC) ) / n / (n-1)
    varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))

    muX = np.dot(np.dot(bone.T, K), bone) / n / (n-1)
    muY = np.dot(np.dot(bone.T, L), bone) / n / (n-1)

    mHSIC = (1 + muX * muY - muX - muY) / n

    al = mHSIC**2 / varHSIC
    bet = varHSIC*n / mHSIC

    thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]

    return testStat, thresh

def hsic_matrix(data, n_sample=False, frac_sample=False):
    """Obtains all coefficient values related to hsic as a correlation coefficient array.
    Calculated coefficients are saved as instances.
    data: (numpy or pandas) A data frame that contains all explanatory and objective variables
    n_sample : (int) How much random sampling to do. False if not.
    If a numerical value is entered, sampling is performed using that number of rows.
    frac_sample: [0 ~ 1] (float) Sampled as a percentage of the number of rows. Not used at the same time as n_sample.
    """

    data = np.array(data)
    data = data[~np.isnan(data).any(axis=1),:]  # Delete rows contain missing values
    # Sampling when n_sample contains a numerical value
    # Both definitions
    if n_sample and frac_sample:
        raise ValueError('n_sample and frac_sample don`t using both')
    elif not n_sample and frac_sample:
        # n_sample=False, frac_sample=int
        data = data.sample(frac=frac_sample, replace=True)
    elif n_sample and not frac_sample:
        # n_sample=int, frac_sample=False
        data = data.sample(n=n_sample, replace=True)
    # else is pass

    n_col = data.shape[1]
    hsic = []
    hsic_append = hsic.append
    hsic_thresh = []
    hsic_thresh_append = hsic_thresh.append
    for i in range(n_col):
        hsic_row = []  # list to store one line of hsic
        hsic_row_append = hsic_row.append
        hsic_thresh_row = []
        hsic_thresh_row_append = hsic_thresh_row.append
        for j in range(n_col):
            # Calculate only one of them because it is symmetric
            if i>j:
                hsic_row_append(0.0)
                hsic_thresh_row_append(0.0)
            else:
                hsic_value, hsic_thresh_value = hsic_gam(data[:,[i]], data[:,[j]])
                hsic_row_append(hsic_value)
                hsic_thresh_row_append(hsic_thresh_value)
        hsic_append(hsic_row)
        hsic_thresh_append(hsic_thresh_row)

    hsic = np.array(hsic)
    hsic_thresh = np.array(hsic_thresh)
    # Create a array by copying the symmetry of the upper triangular array → transpose → add both → Remove physique components 1 times
    hsic_array = hsic + hsic.T - np.diag(np.diag(hsic, k=0))
    hsic_thresh_array = hsic_thresh + hsic_thresh.T + - np.diag(np.diag(hsic_thresh, k=0))
    return hsic_array, hsic_thresh_array