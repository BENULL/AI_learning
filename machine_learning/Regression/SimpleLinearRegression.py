# coding = utf-8
import numpy as np

class SimpleLinearRegression:

    def __init__(self):
        """初始化Simple Linear Regression模型"""
        self.a_ = None
        self.b_ = None

    def fit(self,x_train,y_train):

        """根据训练数据集x_train, y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        self.a_ = (x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot(x_train - x_mean)
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self,x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self,x_single):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.a_ * x_single + self.b_

    def mean_squared_error(self,y_true,y_predict):
        """计算y_true和y_predict之间的MSE"""

        assert len(y_true) == len(y_predict), \
            "the size of y_true must be equal to the size of y_predict"

        return np.sum((y_true - y_predict) ** 2) / len(y_true)

    def r2_score(self,y_test,y_predict):
        """计算y_true和y_predict之间的R Square"""
        return 1 - self.mean_squared_error(y_true, y_predict) / np.var(y_true)

    def score(self, x_test, y_test):
        """根据测试数据集 x_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(x_test)
        return self.r2_score(y_test, y_predict)

if __name__ == '__main__':
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([1., 3., 2., 3., 5.])
    reg2 = SimpleLinearRegression()
    reg2.fit(x, y)
    x_predict = 6
    print(reg2.predict(np.array([x_predict])))



