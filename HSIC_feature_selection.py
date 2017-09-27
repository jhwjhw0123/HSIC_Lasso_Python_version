#!/usr/bin/env python3
import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages')
import math
import random
import numpy as np
from numpy import matlib
from sklearn import linear_model
from scipy import optimize
#import sklearn-based class to perform pipeline process
from sklearn.base import BaseEstimator, TransformerMixin

class HSIC_Lasso(BaseEstimator,TransformerMixin):
    """Data in the form of [nData * nDim], where nDim stands for the number of features.
       Fitting the HSIC Lasso will get the coefficients of different features, and the only the feature that are selected
       will have a coefficient that grater than 0.
       Initialization Attributes:
            para_lambda: the parameter that controls the trade-off between HSIC criteria and sparsity, larger para_lambda
                         will lead to more coefficients as 0
            sigma_mode: A string that controls the mode to select sigma value in the Gaussian kernel.
                        If it is 'auto', then the HSIC kernel will compute the sigma value by itself.
                        If it is 'input', then the HSIC kernel will use the input sigma value.
            sigma: only valid when mode = 'input', give the inpute value
            feature_mode: A string that indicate the mode for how to select the features.
                        If it is 'top_select', then 'para_lambda' will be ignored and the model will use a small lambda to select feature
                        If it is 'lambda_select', then the routine use the given lambda to select model
    """
    def __init__(self,para_lambda,model_mode,sigma_mode,feature_mode,feature_num=1,sigma=0.01):
        """
        :param para_lambda: The trad-off parameter lambda
        :param model_mode: Control 'classification' or 'regression' mode
        :param sigma_mode: Whether to compute the sigma value or to manually input this
        :param feature_mode: choose to select top n features, or to return all the non-zero features for the given lambda
        :param feature_num: Only valid if feature_mode = 'top_select', specify how many features to return
        :param sigma: Only valid if sigma_mode = 'input', fix the manually-input sigma value
        """
        self.para_lambda = para_lambda
        self.model_mode = model_mode
        self.sigma_mode = sigma_mode
        self.sigma = sigma
        self.alpha = np.zeros([1])
        self.feature_mode = feature_mode
        self.feature_num = feature_num

    def GaussianKernel(self,x_left,x_right,sigma):
        xAmount = np.shape(x_left)[0]
        yAmount = np.shape(x_right)[0]
        nDim = np.shape(x_left)[1]
        x_left = x_left.reshape([xAmount, nDim])
        x_right = x_right.reshape([yAmount, nDim])
        divident = 2 * math.pow(sigma, 2)
        # vectorized kernel
        kernel_left = np.sum(np.power(x_left,2),axis=1)
        kernel_right = np.sum(np.power(x_right,2),axis=1)
        kernel_left = kernel_left.reshape([kernel_left.shape[0], 1])
        kernel_right = kernel_right.reshape([kernel_right.shape[0], 1])
        Kernel = np.matlib.repmat(kernel_left,1,yAmount) + np.matlib.repmat(kernel_right.T,xAmount,1) \
                 - 2*np.matmul(x_left,x_right.T)
        Kernel = np.divide(Kernel, divident)

        return Kernel

    def Delta_kernel(self,y_data):
        """
        This kernel only accept data that have the same ammount and with dimension 1
        Specifically for the classification output
        :param y_data: [nData * 1] data specifically for the classification output
        :return: [nData * nData] Delta kernel
        """
        nData = y_data.shape[0]
        if (y_data.shape[1]!=1):
            raise ValueError('The input data for delta kernel must be in dimension 1!')
        y_data = np.reshape(y_data,[nData,1])
        ide_ele, label_counts = np.unique(y_data,return_counts=True)
        # ref_Kernel = np.zeros([nData,nData])
        # for row in range(nData):
        #     for col in range(nData):
        #         if y_data[row]==y_data[col]:
        #             ref_Kernel[row,col] = 1/label_counts[y_data[col]]
        class_n_y = 1/label_counts
        class_label_count = np.reshape(y_data,[nData]).astype(float)
        for ind in range(class_n_y.shape[0]):
            this_ele = ide_ele[ind]
            this_ele_inverse_count = class_n_y[ind]
            class_label_count[class_label_count==this_ele] = this_ele_inverse_count
        class_label_count = np.reshape(class_label_count,[nData,1])
        y_data = np.reshape(y_data,[nData,1])
        class_mat = matlib.repmat(y_data,1,nData)
        comp_mat = np.equal(class_mat,class_mat.T)
        comp_mat[comp_mat==True] = 1
        comp_mat[comp_mat==False] = 0
        Kernel = np.multiply(matlib.repmat(class_label_count,1,nData),comp_mat)

        return Kernel


    def sigma_calculation(self,x_left,x_right):
        # This is quoted from the HSIC Matlab program from the original author
        # Currently cannot find justification for this method, contribution on this will be warmly welcomed
        if x_left.shape[0] > x_right.shape[0]:
            Data_amount = x_right.shape[0]
        else:
            Data_amount = x_left.shape[0]
        if Data_amount > 500:
            random_index = random.sample(range(Data_amount), 500)
            Data_amount = 500
            # Only keep up to 500 examples to avoid computational cost
        else:
            random_index = random.sample(range(Data_amount), Data_amount)
        x_left_med = x_left[random_index]
        x_right_med = x_right[random_index]
        G = np.sum(np.multiply(x_left_med, x_right_med), axis=1)
        G = np.reshape(G, [G.shape[0], 1])
        Q = np.matlib.repmat(G, 1, Data_amount)
        R = np.matlib.repmat(G.T, Data_amount, 1)
        dists = Q + R - 2 * np.dot(x_left_med, x_right_med.T)
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, [Data_amount * Data_amount, 1])
        pos_dists = dists[dists > 0]
        sigma = math.sqrt(0.5 * np.median(pos_dists))

        return sigma

    def fit(self,data_x,data_y):
        #data_x is in [nData*nDim]
        nData = data_x.shape[0]
        nDim = data_x.shape[1]
        if (nDim<self.feature_num and self.feature_mode == 'top_select'):
            raise ValueError('Number of select features cannot be greater than the dimension of original data')
        # Reshape Y to get fit the shape of [nData x 1]
        Y = data_y.reshape([nData, 1])
        # centering Matrix
        H = np.eye(nData) - np.divide(np.ones((nData, nData)), nData)
        # Calculate LH for y kernel
        if self.sigma_mode=='auto':
            sigma = self.sigma_calculation(Y,Y)
        elif self.sigma_mode == 'input':
            sigma = self.sigma
        else:
            raise ValueError('Input mode unrecognized!')
        if math.isnan(sigma) == True:
            sigma = 1e-3  # Manually assign a non-zero number
        if self.model_mode == 'regression':
            L = self.GaussianKernel(Y, Y,sigma)
        elif self.model_mode == 'classification':
            L = self.Delta_kernel(Y)
        else:
            raise ValueError('Model mode not recognized! Could only be regression or classification!')
        LH = np.dot(np.dot(H, L), H)
        # Vectorize the matrices to obtain a standard non-negative Lasso form
        LH_vec = LH.reshape([LH.shape[0] * LH.shape[1]])
        # Calculate KH dimension-wise
        KH_vec = []
        for cDim in range(nDim):
            This_X = data_x[:, cDim]
            This_X = This_X.reshape(This_X.shape[0], 1)
            #Choose how to get/use sigma
            if self.sigma_mode == 'auto':
                sigma = self.sigma_calculation(This_X, This_X)
            elif self.sigma_mode == 'input':
                sigma = self.sigma
            else:
                raise ValueError('Input mode unrecognized!')
            # print(sigma)
            if math.isnan(sigma) == True:
                sigma  = 1e-3       #Manually assign a non-zero number
            This_K = self.GaussianKernel(This_X, This_X,sigma)
            This_KH = np.dot(np.dot(H, This_K), H)
            This_KH_vec = This_KH.reshape([This_KH.shape[0] * This_KH.shape[1]])
            KH_vec.append(This_KH_vec)
        KH_vec = np.asarray(KH_vec).T
        # Key variables:
        #   max_iter: control the maximum time of iteration
        #   positive: This must be true because the HSIC is a non-negative Lasso
        #   tol: This controls the stopping criteria
        #   normalize: This controls whether to normalize the data before using it
        #   fit_intercept: controls whether to 'centralize' the data before computing
        if self.feature_mode == 'top_select':
            self.alpha, _ = optimize.nnls(KH_vec,LH_vec)

        elif self.feature_mode == 'lambda_select':
            para_lambda = self.para_lambda
            hsic_target = linear_model.Lasso(alpha=para_lambda, fit_intercept=False,
                                             max_iter=1000, normalize=False, positive=True, precompute=False,
                                             random_state=None, tol=1e-3, warm_start=False)
            # Two variables: K_vec and L_vec
            hsic_target.fit(KH_vec, LH_vec)

            self.alpha = hsic_target.coef_
        else:
            raise ValueError('Feature selection mode unrecognized!')


    def data_feature_select(self,data_x):
        if self.feature_mode == 'top_select':
            non_zero_ind = np.nonzero(self.alpha)[0]
            non_zero_ind_len = non_zero_ind.shape[0]
            if non_zero_ind_len>=self.feature_num:
                sort_ind = np.flip(np.argsort(self.alpha),axis=0)
                select_feature_ind = sort_ind[0:self.feature_num]
                select_feature_ind = np.sort(select_feature_ind)
            else:
                date_dim = data_x.shape[1]
                random_fill_len = self.feature_num - non_zero_ind_len
                diff_index = np.setdiff1d(np.arange(date_dim),non_zero_ind)
                random_fill_ind = np.random.permutation(diff_index)[0:random_fill_len]
                select_feature_ind = np.concatenate((non_zero_ind,random_fill_ind),axis=0)
        elif self.feature_mode == 'lambda_select':
            select_feature_ind = np.flatnonzero(self.alpha)
        else:
            raise ValueError('Feature selection mode unrecognized!')

        data_new = data_x[:, select_feature_ind]


        return data_new, select_feature_ind

# x_data = np.array([[1,5,9],[10,48,8],[100,420,10],[110,430,89]])
# x_data_test = np.array([[2,3,55],[35,67,6],[116,233,45]])
# y_data = np.array([[5],[20],[200],[220]])
# HSIC_feature_selection = HSIC_Lasso(para_lambda=1e-3,model_mode='regression',sigma_mode='auto',sigma=0.01,feature_mode="top_select",feature_num=2)
# HSIC_feature_selection.fit(data_x=x_data,data_y=y_data)
# print(HSIC_feature_selection.alpha)
# x_data_test_selected, feature_index = HSIC_feature_selection.data_feature_select(data_x=x_data_test)
# print(feature_index)
# print(feature_index)