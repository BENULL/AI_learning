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

    def fit_gd(self,X_train,y_train,eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta,X_b,j):
            try:
                return np.sum((y-X_b.dot(theta))**2)/len(y)
            except:
                return float('inf')

        def dJ(theta,X_b,y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b)

        def gradient_descent(X_b,y,initial_theta,eta,n_iters=1e4,epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter<n_iters:
                gradient = dJ(theta,X_b,y)
                last_theta = theta
                theta -= eta*gradient
                if abs(J(theta,X_b,y)-J(last_theta,X_b,y))<epsilon:
                    break
                cur_iter+=1
            return theta

        X_b = np.hstack((np.ones((len(X_train),1)),X_train))
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b,y_train,initial_theta,eta,n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_sgd(self,X_train,y_train,n_iters=5,t0=5,t1=50):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta, n_iters, t0=5, t1=50):

            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)
            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta,X_b_new[i],y_new[i])
                    theta -= learning_rate(cur_iter*m+i)*gradient
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)

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
    from sklearn import preprocessing

    boston = datasets.load_boston()

    X = boston.data
    y = boston.target

    X = X[y < 50.0]
    y = y[y < 50.0]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,random_state=666)
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    reg = LinearRegression()
    reg.fit_sgd(X_train, y_train)
    print(reg.coef_)
    print(reg.intercept_)
