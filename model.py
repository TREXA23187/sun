import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
from data_process import get_data_split


# 6.模型开发与比较（图表）
# dnn
def mlp_func(x, y, x_predict, y_label):
    print("mlp...")
    mlp = MLPRegressor(hidden_layer_sizes=(100,), random_state=42)
    # print(cross_val_score(mlp, x, y.ravel(), cv=5).mean())
    mlp.fit(x, y)
    mlp_predictions = mlp.predict(x_predict)
    print('============MLP Result============')
    print_score(y_label, mlp_predictions)


# 决策树
def dt_func(x, y, x_predict, y_label):
    print("dt...")
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(x, y)
    dt_predictions = dt.predict(x_predict)
    print('============DT Result============')
    print_score(y_label, dt_predictions)


# 随机森林
def rf_func(x, y, x_predict, y_label):
    print("rf...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x, y)
    rf_predictions = rf.predict(x_predict)
    print('============RF Result============')
    print_score(y_label, rf_predictions)

    return rf


# 7.评估模型结果指标
def print_score(label, predictions):
    print('R2 score:', r2_score(label.ravel(), predictions))
    print('Root mean square error', np.sqrt(mean_squared_error(label.ravel(), predictions)))
    print('Mean absolute error', mean_absolute_error(label.ravel(), predictions))
    print('MAPE:', mean_absolute_percentage_error(label.ravel(), predictions) * 100)
    print('\n')


def plot_importance(model, x_label):
    importance = list(model.feature_importances_)
    feature_importance = [(feature, round(importance, 2)) for feature, importance in zip(x_label, importance)]
    # feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

    # [print('Variable:{:20} importance: {}'.format(*pair)) for pair in feature_importance]

    f, ax = plt.subplots(figsize=(10, 7))
    x_values = list(range(len(importance)))

    ax.bar(x_values, importance, orientation='vertical')
    ax.set_xticks(x_values, x_label, rotation=40)
    ax.set_ylabel('Importance')
    ax.set_xlabel('Variable')
    ax.set_title('Variable Importance')


def modelling():
    print("modelling...")
    selected_columns = ['Latitude', 'Humidity', 'Temp', 'Wind',
                        'Visibility', 'Pressure', 'Cloud', 'Location_Grissom',
                        'Location_Hill Weber', 'Location_JDMT', 'Location_Kahului',
                        'Location_MNANG', 'Location_Malmstrom', 'Location_March AFB',
                        'Location_Offutt', 'Location_Peterson', 'Location_Travis',
                        'Location_USAFA', 'Season_Spring', 'Season_Summer', 'Season_Winter',
                        'sine_mon', 'cos_mon', 'sine_hr', 'cos_hr']
    x_train, x_test, y_train, y_test = get_data_split()

    mlp_func(x_train, y_train, x_test, y_test)
    dt_func(x_train, y_train, x_test, y_test)
    rf = rf_func(x_train, y_train, x_test, y_test)

    plot_importance(rf, selected_columns)
