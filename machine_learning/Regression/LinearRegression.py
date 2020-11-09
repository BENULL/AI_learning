# coding = utf-8
import numpy as np


class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack((np.ones((len(X_train), 1)), X_train))

        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn import model_selection

    boston = datasets.load_boston()

    X = boston.data
    y = boston.target

    X = X[y < 50.0]
    y = y[y < 50.0]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
    reg = LinearRegression()
    reg.fit_normal(X_train, y_train)
    print(reg.coef_)
    print(reg.intercept_)
