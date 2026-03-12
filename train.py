import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#加载数据集
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
print(train_data.shape)
#数据清洗
    #考虑到本题目的评测是基于结果的对数差值进行RMSE,所以对训练集价格取对数
train_data["SalePrice"] = np.log1p(train_data["SalePrice"])
    #记录并去掉ID行
train_ID=train_data["Id"]
test_ID=test_data["Id"]
train_data.drop("Id",axis=1,inplace=True)
test_data.drop("Id",axis=1,inplace=True)
    #在这个特征上发现两个偏离值去除
# x=train_data["GrLivArea"]
# y=train_data["SalePrice"]
# plt.scatter(x,y)
# plt.show()
train_data.drop(train_data[(train_data["SalePrice"]<13)&(train_data["GrLivArea"]>4000)].index,inplace=True)
# print(train_data.shape)
y_train=train_data["SalePrice"].values
#数据清洗与缺失值补全
num_train=train_data.shape[0]
num_test=test_data.shape[0]
    #合并数据便于处理
all_data=pd.concat([train_data,test_data],axis=0).reset_index(drop=True)
all_data.drop("SalePrice",axis=1,inplace=True)
    #列出缺失值数量，从大到小
all_data_na=(all_data.isnull().sum()/len(all_data))*100
all_data_na=all_data_na.drop(all_data_na[(all_data_na==0)].index).sort_values(ascending=False)
missing_data=pd.DataFrame({"missing_radio":all_data_na})
print(missing_data)
    #填充NA值
cols_fillna_none = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType', 'MSSubClass'
]
all_data[cols_fillna_none]=all_data[cols_fillna_none].fillna("None")

cols_fillna_zero = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]
all_data[cols_fillna_zero]=all_data[cols_fillna_zero].fillna(0)

all_data["LotFrontage"]=all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x:x.fillna(x.median()))

cols_fillna_mode = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional']
for col in cols_fillna_mode:
    all_data[col]=all_data[col].fillna(all_data[col].mode()[0])

all_data = all_data.drop(['Utilities'], axis=1)

# remaining_na=all_data.isnull().sum().sum()
# print(remaining_na)
    #进行数据类型转化
print(all_data.info())
all_data["MSSubClass"]=all_data["MSSubClass"].astype(str)
all_data["OverallCond"]=all_data["OverallCond"].astype(str)
all_data["MoSold"]=all_data["MoSold"].astype(str)
all_data["YrSold"]=all_data["YrSold"].astype(str)
    #编码处理
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))
    #特征创造
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    #将不均匀数据均匀化
from scipy.stats import skew
from scipy.special import boxcox1p
numeric_features=all_data.dtypes[all_data.dtypes!=object].index
skewed_features=all_data[numeric_features].apply(lambda x:skew(x.dropna())).sort_values(ascending=False)
print("数值分析")
skewness=pd.DataFrame({"skewness":skewed_features})
print(skewness)
skewness=skewness[abs(skewness)>0.75]
skewed_features=skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
    #独热编码
all_data=pd.get_dummies(all_data)
print(f"最终特征工程完成后的数据形状: {all_data.shape}")
    #重新拆分数据集
train_data=all_data[:num_train]
test_data=all_data[num_train:]
#构建模型
