# 决策树回归Python训练代码
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import joblib

# 加载数据
data = pd.read_csv(r'C:\Users\FES\Desktop\gongjingdata.csv')
# 随机打乱数据集顺序
data = data.sample(frac=1).reset_index(drop=True)
X = data.drop(data.columns[-1], axis=1)
Y = data[data.columns[-1]]

# 划分训练集和测试集(80% : 20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=23)

# 定义参数网格
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [1, 2, 3, 4]
}

# 实例化决策树回归模型
dtr = DecisionTreeRegressor(random_state=23)

# 进行网格搜索
grid_search = GridSearchCV(estimator=dtr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, Y_train)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)

# 使用最佳参数训练模型
best_dtr = grid_search.best_estimator_

#保存训练好的模型
joblib.dump(best_dtr, 'decision_tree_model.pkl')

# 对测试集进行预测
Y_pred = best_dtr.predict(X_test)

# 计算均方误差（mse）和相关系数（R2）
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("均方误差（MSE）：", mse)
print("相关系数（R2）:", r2)

# 决策树可视化
# 获取特征名称
feature_names = X.columns  # 获取特征的列名
plt.figure(figsize=(20,10))
plot_tree(best_dtr, filled=True, rounded=True, feature_names=feature_names)
plt.show()

# 结果可视化
plt.scatter(Y_test, Y_pred, color='blue', alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Decision Tree Regression: Actual vs Predicted")
plt.show()