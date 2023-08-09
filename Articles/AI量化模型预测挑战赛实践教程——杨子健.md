---
Title: AI量化模型预测挑战赛实践教程
Date now : 2023-08-09 32 WEEK 
Modification date: 星期三 9日 八月 2023 18:59:51
---
### 提问总结：[我的机器学习之路(学到很多)](https://mp.weixin.qq.com/s/2-V1kFbSzi3Z5UJ7GV_WBw)
#### 问题：下列哪种方法可以用来缓解过拟合的产生？
- A.增加更多的特征
- B.正则化
- C.增加模型的复杂度
- D.以上都是
解释：
- A.增加更多的特征：增加更多特征可能会导致模型变得更加复杂。在某些情况下，引入新的特征可能有助于提高模型的性能和泛化能力，但在其他情况下，增加特征可能会导致模型过拟合。如果增加的特征对于预测任务并没有实质性的帮助，可能会让模型过度关注训练数据的噪声和细节，从而增加过拟合的风险。
- B. 正则化：正则化是缓解过拟合的一种有效方法。它通过在模型的损失函数中增加惩罚项，限制模型参数的大小，从而防止模型过度拟合训练数据。L1正则化和L2正则化是常用的正则化方法，它们通过增加模型参数的L1范数和L2范数惩罚项来实现。这样可以使得模型的某些特征权重变得更小或者为零，从而简化模型并减少过拟合的可能性。
- C. 增加模型的复杂度：增加模型的复杂度通常会增加模型在训练数据上的拟合能力，但也会增加过拟合的风险。当模型过于复杂时，它可能会过度学习训练数据的细节和噪声，而无法很好地泛化到新数据。因此，盲目增加模型的复杂度并不是一个好的缓解过拟合的方法。
- 故A、C、D不是正确选项，答案选B
#### 问题：在回归问题中，常用的评估指标有哪些？（多选）
- A. 均方误差 
- B. 平均绝对误差
- C. 召回率 
- D. F1 分数
解释：
- A：指预测值和平均值差异的平方的平均值
- B：指预测值和平均值差异的绝对值的平均值
- C：召回率分类模型在正例样本中正确预测的比例
- D：F1分数用于分类，不是回归。F1分数指精确率和召回率最高的情况，F1 分数是精确率和召回率的调和平均数，它综合了精确率和召回率两个指标，能够更好地评估分类模型的性能。
#### 问题：如何衡量特征和目标之间的非线性关系？
- A、相关系数
- B、互信息
- C、协方差
- D、最大信息系数
[解释](https://blog.csdn.net/qq_27586341/article/details/90603140)
- A.相关系数：用于度量线性相关分析。
- B.互信息：与相关性不同，其不依赖于数据序列，而是数据分布。在给定一个特征的情况下可以提供多少信息量
- C.协方差：衡量数据两个维度的关系（正相关和负相关）
- D.最大信息系数：最大的基于信息的非参数性探索，衡量变量的关联程度（线性或者非线性）。
#### 问题一：什么是正则化，请解释L1范数和L2范数的区别？
L1范数是各分量绝对值的和，L2范数是各分量平方和的算术平方根；
对于矩阵而言，L1范数是所有矩阵列向量绝对值之和的最大值，L2范数是矩阵左乘自身转置得到的矩阵，其最大特征值的算术平方根。区别在于，L1范数会产生稀疏的解，下降速度一般比L2范数快。
正则化即对不希望得到的结果施加惩罚，以使优化过程趋于期望目标，从而减少最小化训练误差中过拟合的风险
L1范数、L2范数都是属于Lp范数常用的正则化项，其中L1倾向于将非零向量分量个数减少，L2倾向于将非零向量分量个数尽量稠密
### 基本流程
![](photo/Pasted%20image%2020230809100405.png)
### 导入模块：可尝试使用百度飞桨算力或[本地配置环境](https://datawhaler.feishu.cn/docx/EOypdKkujom8THxWkGZc3F4qn8c#doxcnNetmKRr155mpGfCaGBxRJg)
```Python
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, mean_squared_log_error
import tqdm, sys, os, gc, argparse, warnings
warnings.filterwarnings('ignore')
```
### 数据探索：了解数据集，变量间以及与预测值之间的关系
- 买价指的是买方愿意为一项股票/资产支付的最高价格。 
- 卖价指的是卖方愿意接受的一项股票/资产的最低价格。
- 这两个价格之间的差异被称为点差；点差越小，该品种的流动性越高。
--- 
### 特征工程
-   在模型基本选定之后，接下来的要做的就是细致的特征工程，模型与特征是相辅相成的，此处我们将模型与特征工程当做一个整体进行处理。对于设计的模型我们希望它可以充分吸收数据并从数据集中自动挖掘出与我们标签相关的信息，从而能更好地对我们的测试数据进行预测，但从目前模型的发展情况来看，暂时还没有哪种模型可以自动化地对数据进行充分的挖掘，因而我们需要通过人为的方式对数据进行处理，包括特征预处理、组合特征的构建、特征的筛选等等，在模型数据处理的弱势区域对其进行帮助，从而使得我们模型可以获得更好的效果。换言之，特征工程就是在帮助模型学习，在模型学习不好的地方或者难以学习的地方，采用特征工程的方式帮助其学习，通过人为筛选、人为构建组合特征让模型原本很难学好的东西可以更加轻易的学习从而拿到更好的效果。在后续的内容中，我们会针对目前在大数据竞赛圈和工业界表格数据问题上最为流行的梯度提升树模型进行探讨，先介绍针对梯度提升树可以采用的通用特征工程方案以及在特定领域的许多业务特征。
```python
# 为了保证时间顺序的一致性，故进行排序 
train_df = train_df.sort_values(['file','time']) test_df = test_df.sort_values(['file','time']) 
# 当前时间特征 
# 围绕买卖价格和买卖量进行构建 
# 暂时只构建买一卖一和买二卖二相关特征，进行优化时可以加上其余买卖信息 

# 计算买一价和卖一价的加权平均作为新特征'wap1'，加权平均的计算方式是：(买一价 * 买一量 + 卖一价 * 卖一量) / (买一量 + 卖一量)
train_df['wap1'] = (train_df['n_bid1'] * train_df['n_bsize1'] + train_df['n_ask1'] * train_df['n_asize1']) / (train_df['n_bsize1'] + train_df['n_asize1'])
# 计算买二价和卖二价的加权平均作为新特征'wap2'，加权平均的计算方式同样是：(买二价 * 买二量 + 卖二价 * 卖二量) / (买二量 + 卖二量)
train_df['wap2'] = (train_df['n_bid2'] * train_df['n_bsize2'] + train_df['n_ask2'] * train_df['n_asize2']) / (train_df['n_bsize2'] + train_df['n_asize2'])
# test 同样

# 计算'wap1'和'wap2'之间的差值的绝对值作为新特征'wap_balance'
train_df['wap_balance'] = abs(train_df['wap1'] - train_df['wap2'])
# 计算买一价和买二价之间的差值作为新特征'bid_spread'
train_df['bid_spread'] = train_df['n_bid1'] - train_df['n_bid2']
# 计算卖一价和卖二价之间的差值作为新特征'ask_spread'
train_df['ask_spread'] = train_df['n_ask1'] - train_df['n_ask2']
# 计算买一量、买二量、卖一量和卖二量之和作为新特征'total_volume'
train_df['total_volume'] = (train_df['n_asize1'] + train_df['n_asize2']) + (train_df['n_bsize1'] + train_df['n_bsize2'])
# 计算买一量、买二量、卖一量和卖二量之间的差值的绝对值作为新特征'volume_imbalance'
train_df['volume_imbalance'] = abs((train_df['n_asize1'] + train_df['n_asize2']) - (train_df['n_bsize1'] + train_df['n_bsize2']))
```
- 加入三四五信息的版本
![](photo/Pasted%20image%2020230809130103.png)
```python
def calculate_wap(df, i):
    df[f'wap{i}'] = (df[f'n_bid{i}']*df[f'n_bsize{i}'] + df[f'n_ask{i}']*df[f'n_asize{i}'])/(df[f'n_bsize{i}'] + df[f'n_asize{i}']) 
    df[f'wap{i+1}'] = (df[f'n_bid{i+1}']*df[f'n_bsize{i+1}'] + df[f'n_ask{i+1}']*df[f'n_asize{i+1}'])/(df[f'n_bsize{i+1}'] + df[f'n_asize{i+1}']) 
    
    df[f'wap_balance{i}_{i+1}'] = abs(df[f'wap{i}'] - df[f'wap{i+1}']) 
    df[f'price_spread{i}_{i+1}'] = (df[f'n_ask{i}'] - df[f'n_bid{i}']) / ((df[f'n_ask{i}'] + df[f'n_bid{i}'])/2)

	df[f'bid_spread{i}_{i+1}'] = df[f'n_bid{i}'] - df[f'n_bid{i+1}'] 
	df[f'ask_spread{i}_{i+1}'] = df[f'n_ask{i}'] - df[f'n_ask{i+1}'] 

	df[f'total_volume{i}_{i+1}'] = (df[f'n_asize{i}'] + df[f'n_asize{i+1}']) + (df[f'n_bsize{i}'] + df[f'n_bsize{i+1}']) 
	df[f'volume_imbalance{i}_{i+1}'] = abs((df[f'n_asize{i}'] + df[f'n_asize{i+1}']) - df[f'n_bsize{i}'] + df[f'n_bsize{i+1}']))
	return df

def feature_engineering(df):
	for i in range (1,5):
	    df = calculate_wa(df, i)

		# 历史平移 
		# 获取历史信息 
		for val in [f'wap{i}',f'wap{i+1}',f'wap_balance{i}_{i+1}',f'price_spread{i}_{i+1}',
		f'bid_spread{i}_{i+1}',f'ask_spread{i}_{i+1}',f'total_volume{i}_{i+1}',f'volume_imbalance{i}_{i+1}']:
			for loc in [1,5,10,20,40,60]: 
				df[f'file_{val}_shift{loc}'] = df.groupby(['file'])[val].shift(loc)

		# 差分特征 
		# 获取与历史数据的增长关系 
		for val in [f'wap{i}',f'wap{i+1}',f'wap_balance{i}_{i+1}',f'price_spread{i}_{i+1}',
		f'bid_spread{i}_{i+1}',f'ask_spread{i}_{i+1}',f'total_volume{i}_{i+1}',f'volume_imbalance{i}_{i+1}']:
			for loc in [1,5,10,20,40,60]: 
				df[f'file_{val}_diff{loc}'] = df.groupby(['file'])[val].diff(loc) 

		# 窗口统计 
		# 获取历史信息分布变化信息 
		# 可以尝试更多窗口大小已经统计方式，如min、max、median等 
		for val in [f'wap{i}',f'wap{i+1}',f'wap_balance{i}_{i+1}',f'price_spread{i}_{i+1}',
		f'bid_spread{i}_{i+1}',f'ask_spread{i}_{i+1}',f'total_volume{i}_{i+1}',f'volume_imbalance{i}_{i+1}']:
			df[f'file_{val}_win7_mean'] = df.groupby(['file'])[val].transform(lambda x: x.rolling(window=7, min_periods=3).mean()) 
			df[f'file_{val}_win7_std'] = df.groupby(['file'])[val].transform(lambda x: x.rolling(window=7, min_periods=3).std()) 
			test_df[f'file_{val}_win7_mean'] = test_df.groupby(['file'])[val].transform(lambda x: x.rolling(window=7, min_periods=3).mean()) 
			test_df[f'file_{val}_win7_std'] = test_df.groupby(['file'])[val].transform(lambda x: x.rolling(window=7, min_periods=3).std())
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)
```
- 当前时间特征：**围绕买卖价格和买卖量进行构建，暂时只构建买一卖一和买二卖二相关特征，进行优化时可以加上其余买卖信息；
- 历史平移特征：**通过历史平移获取上个阶段的信息；
- 差分特征：**可以帮助获取相邻阶段的增长差异，描述数据的涨减变化情况。在此基础上还可以构建相邻数据比值变化、二阶差分等；
- 窗口统计特征：**窗口统计可以构建不同的窗口大小，然后基于窗口范围进统计均值、最大值、最小值、中位数、方差的信息，可以反映最近阶段数据的变化情况。
#### 优化
提取更多特征：** 在数据挖掘比赛中，**特征**总是最终制胜法宝，去思考什么信息可以帮助我们提高预测精准度，然后将其转化为特征输入到模型。
对于本次赛题可以从业务角度构建特征，在量化交易方向中，常说的因子与机器学习中的特征基本一致，**趋势因子、收益波动率因子、买卖压力、同成交量衰减、斜率 价差/深度**，可以围绕成交量、买价和卖价进行构建。也可以从时间序列预测角度构建特征，比如**历史平移特征、差分特征、和窗口统计特征**。
![](photo/Pasted%20image%2020230809130048.png)
- 类别特征
    - 编码方式 ：自然数编码、独热编码、count编码（替代类别特征）、目标编码  
    - 统计方式：count、nunique（宽度）、ratio（偏好）
- 数值特征
    - 交叉统计：行交叉（均值、中位数、最值）、业务交叉构造
    - 离散方式：分桶、二值化（0/1） （提升泛化）
### 特征筛选
```Python
print('过滤异常特征... ')
drop_cols = []  # 用于存储需要删除的异常特征列的列表
# 遍历DataFrame `df` 的每一列
for col in df.columns:
    # 如果列中的唯一值数量为1，说明该列中所有的值都相同，对于建模来说没有意义，将其添加到 `drop_cols` 列表中
    if df[col].nunique() == 1:
        drop_cols.append(col)
    # 计算每列缺失值的比例，如果缺失值比例超过总行数的 95%，也将该列添加到 `drop_cols` 列表中
    if df[col].isnull().sum() / df.shape[0] > 0.95:
        drop_cols.append(col)

print('过滤高相关特征...')
# 定义一个函数用于检测高相关特征
def correlation(data, threshold):
    col_corr = []  # 用于存储高相关特征列的列表
    corr_matrix = data.corr()  # 计算数据的相关系数矩阵
    # 遍历相关系数矩阵中的每个元素
    for i in range(len(corr_matrix)):
        for j in range(i):
            # 如果某个元素的绝对值大于阈值 `threshold`，说明这两个特征之间存在高相关性
            # 将其中一个特征的名称添加到 `col_corr` 列表中
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.append(colname)
    return list(set(col_corr))  # 返回包含所有高相关特征的列名的列表
```
--- 
--- 
### 模型融合
- 机器学习模型（XGBoost、LightGBM 、catboost）
    - 对特征处理要求低
    - 对类别和连续特征友好
    - 缺失值不需要填充
- NN模型 LSTM   CNN    RNN等  
### 模型融合
```Python
def cv_model(clf, train_x, train_y, test_x, clf_name, seed = 2023):
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros([train_x.shape[0], 3])
    test_predict = np.zeros([test_x.shape[0], 3])
    cv_scores = []
    
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        
        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)
            params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class':3,
                'min_child_weight': 6,
                'num_leaves': 2 ** 6,
                'lambda_l2': 10,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.35,
                'seed': 2023,
                'nthread' : 16,
                'verbose' : -1,
            }
            model = clf.train(params, train_matrix, 2000, valid_sets=[train_matrix, valid_matrix],
                              categorical_feature=[], verbose_eval=1000, early_stopping_rounds=100)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
        
        if clf_name == "xgb":
            xgb_params = {
              'booster': 'gbtree', 
              'objective': 'multi:softprob', 指定学习目标， 'multi:softprob'：输出的概率矩阵
              'num_class':3,
              'max_depth': 5, 树的最大深度，越大越复杂，可以用来控制过拟合
              'lambda': 10, L2正则化项
              'subsample': 0.7,样本采样率，随机选择70%样本作为训练集
              'colsample_bytree': 0.7,构造每课树时，列采样率（一般是特征采样率）
              'colsample_bylevel': 0.7,每执行一次分裂，列采样率
              'eta': 0.35, 学习率，减少权重值，防止过拟合
              'tree_method': 'hist',
              'seed': 520, 随机数种子，用于生成可复制结果
              'nthread': 16， 用于并行处理
              }
            train_matrix = clf.DMatrix(trn_x , label=trn_y)
            valid_matrix = clf.DMatrix(val_x , label=val_y)
            test_matrix = clf.DMatrix(test_x)
            
            watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
            
            model = clf.train(xgb_params, train_matrix, num_boost_round=2000, evals=watchlist, verbose_eval=1000, early_stopping_rounds=100) # 当模型100次无优化后停止
            val_pred  = model.predict(valid_matrix)
            test_pred = model.predict(test_matrix)
            
        if clf_name == "cat":
            params = {'learning_rate': 0.35, 'depth': 5, 'bootstrap_type':'Bernoulli','random_seed':2023,
                      'od_type': 'Iter', 'od_wait': 100, 'random_seed': 11, 'allow_writing_files': False,
                      'loss_function': 'MultiClass'}
            
            model = clf(iterations=2000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                      metric_period=1000,
                      use_best_model=True, 
                      cat_features=[],
                      verbose=1)
            
            val_pred  = model.predict_proba(val_x)
            test_pred = model.predict_proba(test_x)
        
        oof[valid_index] = val_pred
        test_predict += test_pred / kf.n_splits
        
        F1_score = f1_score(val_y, np.argmax(val_pred, axis=1), average='macro')
        cv_scores.append(F1_score)
        print(cv_scores)
        
    return oof, test_predict

# 处理train_x和test_x中的NaN值 
train_df = train_df.fillna(0) test_df = test_df.fillna(0) 
# 处理train_x和test_x中的Inf值 
train_df = train_df.replace([np.inf, -np.inf], 0) test_df = test_df.replace([np.inf, -np.inf], 0) 
# 入模特征 cols = [f for f in test_df.columns if f not in ['uuid','time','file']] 
for label in ['label_5','label_10','label_20','label_40','label_60']: 
	print(f'=================== {label} ===================')
# 参考demo,具体对照baseline实践部分调用cv_model函数
# 选择lightgbm模型
lgb_oof, lgb_test = cv_model(lgb, train_df[cols], train_df['label'], test_df[cols], 'lgb')
# 选择xgboost模型
xgb_oof, xgb_test = cv_model(xgb, train_df[cols], train_df['label'], test_df[cols], 'xgb')
# 选择catboost模型
cat_oof, cat_test = cv_model(CatBoostClassifier, train_df[cols], train_df['label'], test_df[cols], 'cat')

# 进行取平均融合
final_test = (lgb_test + xgb_test + cat_test) / 3
```
### 模型训练与验证(进阶)
定义cv_model函数，内部可以选择使用lightgbm、xgboost和catboost模型，可以依次跑完这三个模型，然后将三个模型的结果进行取平均进行融合。
![](photo/Pasted%20image%2020230809125343.png)
另外一种经典融合方式为stacking，stacking是一种分层模型集成框架。以两层为例，第一层由多个基学习器组成，其输入为原始训练集，第二层的模型则是以第一层基学习器的输出作为特征加入训练集进行再训练，从而得到完整的stacking模型。
- **第一层：（类比cv_model函数）**
	划分训练数据为K折（5折为例，每次选择其中四份作为训练集，一份作为验证集）；
	针对各个模型RF、ET、GBDT、XGB，分别进行5次训练，每次训练保留一份样本用作训练时的验证，训练完成后分别对Validation set，Test set进行预测，对于Test set一个模型会对应5个预测结果，将这5个结果取平均；对于Validation set一个模型经过5次交叉验证后，所有验证集数据都含有一个标签。此步骤结束后：**5个验证集（总数相当于训练集全部）在每个模型下分别有一个预测标签，每行数据共有4个标签（4个算法模型），测试集每行数据也拥有四个标签（4个模型分别预测得到的）**
- **第二层：（类比stack_model函数）**
	将训练集中的四个标签外加真实标签当作**五列新的特征作为新的训练集**，选取一个训练模型，根据新的训练集进行训练，然后应用**测试集的四个标签组成的测试集**进行预测作为最终的result。
```Python
def stack_model(oof_1, oof_2, oof_3, predictions_1, predictions_2, predictions_3, y):
    '''
    输入的oof_1, oof_2, oof_3可以对应lgb_oof，xgb_oof，cat_oof
    predictions_1, predictions_2, predictions_3对应lgb_test，xgb_test，cat_test
    '''
    train_stack = pd.concat([oof_1, oof_2, oof_3], axis=1)
    test_stack = pd.concat([predictions_1, predictions_2, predictions_3], axis=1)
    
    oof = np.zeros((train_stack.shape[0],))
    predictions = np.zeros((test_stack.shape[0],))
    scores = []
    
    from sklearn.model_selection import RepeatedKFold
    folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2021)
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, train_stack)): 
        print("fold n°{}".format(fold_+1))
        trn_data, trn_y = train_stack.loc[trn_idx], y[trn_idx]
        val_data, val_y = train_stack.loc[val_idx], y[val_idx]
        
        clf = Ridge(random_state=2021)
        clf.fit(trn_data, trn_y)

        oof[val_idx] = clf.predict(val_data)
        predictions += clf.predict(test_stack) / (5 * 2)
        
        score_single = roc_auc_score(val_y, oof[val_idx])
        scores.append(score_single)
        print(f'{fold_+1}/{5}', score_single)
    print('mean: ',np.mean(scores))
   
    return oof, predictions
```

---
### 验证模型
- 时序验证
![](photo/Pasted%20image%2020230809131814.png)
- 交叉验证
![](photo/Pasted%20image%2020230809131822.png)

### 结果验证
```python
import pandas as pd 
import os 
# 检查并删除'submit'文件夹 
if os.path.exists('./submit'): 
	shutil.rmtree('./submit') 
	print("Removed the 'submit' directory.") 
# 检查并删除'submit.zip'文件 
if os.path.isfile('./submit.zip'): 
	os.remove('./submit.zip') 
	print("Removed the 'submit.zip' file.") 
# 指定输出文件夹路径 
output_dir = './submit' 
# 如果文件夹不存在则创建 
if not os.path.exists(output_dir): 
	os.makedirs(output_dir) 
# 首先按照'file'字段对 dataframe 进行分组 
grouped = test_df.groupby('file') 
# 对于每一个group进行处理 
for file_name, group in grouped: 
# 选择你所需要的列 
	selected_cols = group[['uuid', 'label_5', 'label_10', 'label_20', 'label_40', 'label_60']] 
	# 将其保存为csv文件，file_name作为文件名 
	selected_cols.to_csv(os.path.join(output_dir, f'{file_name}'), index=False) 
_ = !zip -r submit.zip submit/
```

### **官方文档**
[在线版本Baseline与解题思路](https://datawhaler.feishu.cn/docx/EOypdKkujom8THxWkGZc3F4qn8c)
[AI量化模型预测挑战赛:](https://challenge.xfyun.cn/topic/info?type=quantitative-model&ch=ymfk4uU)
[竞赛实践路线分享](https://datawhaler.feishu.cn/docx/EJ2Edl0hXoIWwuxO15CcEj9Wnxn)
[Scikit-Learn保姆教程：](https://mp.weixin.qq.com/s/4NSVh1HniNT4CGakzHxm1w)
[时间序列预测方法总结](https://zhuanlan.zhihu.com/p/67832773)

