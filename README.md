# HSIC_Lasso_Python_version <br />
This is a Python version of HSIC Lasso based on the paper *High-Dimensional Feature Selection by Feature-Wise Kernelized Lasso* by M.Yamada et al. (URL: https://arxiv.org/pdf/1202.0515.pdf). The author of the paper has a published version of this method now, however this program directly reproduced the algorithm from the paper. <br />
Comparing to the work of the original author, the optimization method in this project is different: here I use sk-learn Lasso optimization method to optimze the target function. The kernelization methods in this program is vectorized as possible. <br />
Last Modified: Chen Wang
