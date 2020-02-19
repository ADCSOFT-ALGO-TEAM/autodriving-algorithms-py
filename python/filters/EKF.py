"""
~~~~~~~~~~~~~~~
扩展 Kalman 滤波
~~~~~~~~~~~~~~~

作者: mathzhaoliang@gmail.com

这个模块主要实现了 'ExtendedKalmanFilter' 这个类, 在使用时需要注意如下事项:

1. 在类初始化后, 滤波开始前应当手动设置如下的量:
   x0 : 初始状态向量.
   F : 模型状态转移矩阵.
   Q : 过程噪声协方差矩阵
   R : 观测噪声协方差矩阵

2. 在扩展 Kalman 滤波中, 必须实现两个函数 'HJacobian' 和 'Hx', 它们的
   的作用是针对当前的状态向量 x, 返回其对应的状态转移函数和观测函数在 x 处
   的 Jacobi 矩阵.

"""
from copy import deepcopy
import numpy as np


class ExtendedKalmanFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0):
        """
        dim_x : 状态向量维数

        dim_z : 观测向量维数

        dim_u : 控制向量维数

        x : 状态估计向量 (根据不同阶段可能等于先验估计或者后验估计)

        P : 状态估计误差的协方差矩阵 (即 x 的误差的协方差)

        x_prior : 先验估计状态向量, 即 x 在子空间 { z_i, i<k } 上的最佳线性逼近.

        P_prior : 先验估计误差 (x - x_prior) 的协方差矩阵.

        x_post : 后验估计状态向量, 即 x 在子空间 { z_i, i<=k } 上的最佳线性逼近.

        P_post : 后验估计误差 (x - x_post) 的协方差矩阵.

        Q : 过程噪声协方差矩阵.

        R : 观测噪声协方差矩阵.

        F : 模型状态转移矩阵.

        H : 观测函数矩阵.

        y : 更新一步中的残差向量, 即 z_k 中垂直于 { z_i, i<k } 的分量.

        K : Kalman 增益矩阵.

        z : 最近一次更新中使用的观测向量.
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.F = np.eye(dim_x)              # 模型状态转移矩阵
        self.B = 0                          # 控制输入矩阵
        self._x = np.zeros(dim_x)           # 系统状态向量
        self._P = np.eye(dim_x)             # 估计误差协方差矩阵
        self._Q = np.eye(dim_x)             # 过程噪声协方差矩阵
        self._R = np.eye(dim_z)             # 测量噪声协方差矩阵
        self._K = np.zeros((dim_x, dim_z))  # Kalman 增益矩阵

        self._z = np.array([None] * self.dim_z)
        self._y = np.zeros(dim_z)
        self._I = np.eye(dim_x)

        self.x_prior = self._x.copy()
        self.P_prior = self._P.copy()

        self.x_post = self._x.copy()
        self.P_post = self._P.copy()

    def set_model_transition(self, F):
        """
        设置模型状态转移矩阵. 必须形如 numpy.array(dim_x, dim_x)
        """
        F = np.asarray(F)
        if F.shape != (self.dim_x, self.dim_x):
            raise ValueError("Model transition matrix shape error: {}".format(F.shape))
        self.F = F

    def set_process_noise(self, Q):
        """
        设置过程噪声协方差矩阵. 必须形如 numpy.array(dim_x, dim_x).
        """
        Q = np.asarray(Q)
        if Q.shape != (self.dim_x, self.dim_x):
            raise ValueError("Process noise covariance shape error: {}".format(Q.shape))
        self._Q = Q

    def set_measurement_noise(self, R):
        """
        设置观测噪声协方差矩阵. 必须形如 numpy.array(dim_z, dim_z).
        """
        R = np.asarray(R)
        if R.shape != (self.dim_z, self.dim_z):
            raise ValueError("Measurement noise covariance shape error: {}".format(R.shape))
        self._R = R

    def set_control(self, B):
        """
        设置控制矩阵, 可以是一个常数, 也可以是一个 numpy.(dim_x, dim_u) 的
        矩阵, 这里不做形状检查.
        """
        self.B = B

    def set_state(self, x0):
        """
        设置系统初始状态. 必须形如 numpy.array(dim_x).
        """
        x0 = np.asarray(x0)
        if x0.shape != (self.dim_x,):
            raise ValueError("State vector dimension error: {}".format(x0.shape))
        self._x = x0

    def get_state(self):
        return self._x

    def predict_x(self, u=0):
        """
        预测步骤, 只更新状态为先验估计但不更新估计误差.
        """
        self._x = np.dot(self.F, self._x) + np.dot(self.B, u)

    def predict(self, u=0):
        """
        预测步骤. 会更新状态为先验估计并更新估计误差.
        """
        self.predict_x(u)
        self._P = np.dot(self.F, self._P).dot(self.F.T) + self._Q

        self.x_prior = np.copy(self._x)
        self.P_prior = np.copy(self._P)
        return self

    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(),
               residual=np.subtract):
        """
        更新步骤. 将先验估计更新为后验估计, 并更新误差的协方差矩阵.

        z : 更新中使用的观测量.
        HJacobian : 这是一个函数, 对状态 x 返回其 Jacobi 矩阵.
        Hx : 这也是一个函数, 对状态 x 返回观测值的 Jacobi 矩阵.
        R : 你可以在这里传入新的观测噪声变量.
        args : HJacobian 函数中使用的其它参数.
        hx_args : Hx 函数中使用的其它参数.
        residual : 这是一个函数, 用于计算两个观测量的残差. 默认为 np.subtract.
        """
        # 如果测量值为 None 的话则不作更新,
        # 使用之前的预测值.
        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self._x.copy()
            self.P_post = self._P.copy()
            return

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self._R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        # 计算 Kalman 增益矩阵
        H = HJacobian(self._x, *args)
        PHT = np.dot(self._P, H.T)
        S = np.dot(H, PHT) + R
        self._K = PHT.dot(np.linalg.inv(S))

        # 计算更新后的估计值
        hx = Hx(self._x, *hx_args)
        self._y = residual(z, hx)
        self._x += np.dot(self._K, self._y)

        # 计算更新后的误差协方差
        I_KH = self._I - np.dot(self._K, H)
        self._P = np.dot(I_KH, self._P).dot(I_KH.T) + np.dot(self._K, R).dot(self._K.T)

        # 重置 x_post 和 P_post
        self._z = deepcopy(z)
        self.x_post = self._x.copy()
        self.P_post = self._P.copy()

        return self
