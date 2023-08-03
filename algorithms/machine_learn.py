from sklearn.linear_model import LinearRegression

from algorithms.utils import regression_matrics
from feature.analysis_utils import plot_comparison


def linear_regression(X_train, X_test, y_train, y_test, y_label, **kwargs):
    """
    线性回归
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param y_label:
    :return:
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred_lr = regressor.predict(X_test)
    print('Linear Regression:')
    regression_matrics(y_test, y_pred_lr)
    plot_comparison(y_test=y_test, y_pred=y_pred_lr, xlabel="index", ylabel=y_label, dark_mode=True)
    return regressor
