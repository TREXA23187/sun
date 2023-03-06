import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB

import matplotlib.pyplot as plt
from data_process import data_splint_with_feature_engineering, data_splint_without_feature_engineering


# 6.模型开发与比较（图表）
# dnn
def mlp_func(x, y, x_predict, y_label):
    print("mlp...")
    mlp = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
    mlp.fit(x, y)
    mlp_predictions = mlp.predict(x_predict)
    print('============MLP Result============')
    print(cross_val_score(mlp, x, y.ravel(), cv=5, scoring="accuracy").mean())
    print_score(y_label, mlp_predictions)


# 决策树
def dt_func(x, y, x_predict, y_label):
    print("dt...")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(x, y)
    dt_predictions = dt.predict(x_predict)
    print('============DT Result============')
    print(cross_val_score(dt, x, y.ravel(), cv=5, scoring="accuracy").mean())
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


# knn
def knn_func(x, y, x_predict, y_label):
    print("knn...")
    knn = KNeighborsClassifier()
    knn.fit(x, y)
    knn_predictions = knn.predict(x_predict)
    print('============KNN Result============')
    print(cross_val_score(knn, x, y.ravel(), cv=5, scoring='accuracy').mean())
    print_score(y_label, knn_predictions)

    return knn


def nb_func(x, y, x_predict, y_label):
    print("nb...")
    mnb = GaussianNB()
    mnb.fit(x, y)
    mnb_predictions = mnb.predict(x_predict)
    print('============MNB Result============')
    print(mnb.score(x_predict, y_label))
    print_score(y_label, mnb_predictions)

    return mnb


# 7.评估模型结果指标
def print_score(label, predictions):
    # print('R2 score:', r2_score(label.ravel(), predictions))
    print('Root mean square error', np.sqrt(mean_squared_error(label.ravel(), predictions)))
    print('Mean absolute error', mean_absolute_error(label.ravel(), predictions))
    print('MAPE:', mean_absolute_percentage_error(label.ravel(), predictions) * 100)
    print('\n')


def plot_importance(model, x_label):
    importance = list(model.feature_importances_)
    print(importance)
    feature_importance = [(feature, round(importance, 2)) for feature, importance in zip(x_label, importance)]
    # feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

    [print('Variable:{:20} importance: {}'.format(*pair)) for pair in feature_importance]

    f, ax = plt.subplots(figsize=(10, 7))
    x_values = list(range(len(importance)))

    ax.bar(x_values, importance, orientation='vertical')
    ax.set_xticks(x_values, x_label, rotation=40, fontsize=4.5)
    ax.set_ylabel('Importance')
    ax.set_xlabel('Variable', fontsize=8)
    ax.set_title('Variable Importance')


def assemble(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import accuracy_score

    abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
    abc.fit(x_train, y_train)

    print(abc.__class__.__name__, '(DecisionStumps)')
    y_pred = abc.predict(x_train)
    print('\ttrain:', accuracy_score(y_train, y_pred))
    y_pred = abc.predict(x_test)
    print('\ttest:', accuracy_score(y_test, y_pred))


def modelling_1():
    print("modelling 1...")
    x_train, x_test, y_train, y_test = data_splint_without_feature_engineering()

    # dt_func(x_train, y_train, x_test, y_test)
    knn_func(x_train, y_train, x_test, y_test)
    nb_func(x_train, y_train, x_test, y_test)
    # mlp_func(x_train, y_train, x_test, y_test)
    # assemble(x_train, y_train, x_test, y_test)
    # rf = rf_func(x_train, y_train, x_test, y_test)

    # plot_importance(rf, selected_columns)


def modelling_2():
    print("modelling 2...")
    x_train, x_test, y_train, y_test = data_splint_with_feature_engineering()

    # dt_func(x_train, y_train, x_test, y_test)
    knn_func(x_train, y_train, x_test, y_test)
    nb_func(x_train, y_train, x_test, y_test)
    # mlp_func(x_train, y_train, x_test, y_test)
    # assemble(x_train, y_train, x_test, y_test)
    # rf = rf_func(x_train, y_train, x_test, y_test)

    # plot_importance(rf, selected_columns)


if __name__ == '__main__':
    modelling_1()
    modelling_2()
