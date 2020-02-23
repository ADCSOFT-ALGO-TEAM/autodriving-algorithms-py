"""
~~~~~~~~~~~~~~~
扩展 Kalman 滤波
~~~~~~~~~~~~~~~

作者: mathzhaoliang@gmail.com


扩展 Kalman 滤波方程:

    xₖ = f(xₖ₋₁, uₖ₋₁) + wₖ₋₁
    zₖ = h(xₖ₋₁)       + vₖ

其中 xₖ 是 k 时刻的状态向量, zₖ 是 k 时刻的观测向量, wₖ, vₖ 分别是过程噪声和观测噪声.

这个模块需要使用者自己实现一些函数: (名字可以不同, 这里的函数名仅作为示例)

1. 'get_control_input': 此函数对给定的状态 x, 计算对应的控制输入 u.
2. 'Fx(x, u)': 状态转移函数 F.
3. 'JFx(x, u)': 状态转移函数在 (x, u) 处的 Jacobi 矩阵.
4. 'Hx(x)': 状态到观测的转移函数 H.
5. 'JHx': 返回 H(x) 在 x 处的 Jacobi 矩阵.

此外需要使用者手动设置如下属性:

1. 系统状态和观测向量的维数 dim_x, dim_u.
2. 系统初始状态 x0, P0.
3. 协方差矩阵 Q, R.

"""
import numpy as np


class ExtendedKalmanFilter(object):

    def __init__(self, dim_x, dim_z,
                 Q=None, R=None,
                 init_x=None, init_P=None):
        """
        dim_x : 状态向量维数

        dim_z : 观测向量维数

        dim_u : 控制向量维数

        init_x : 初始状态向量估计值

        init_P : 初始状态协方差

        其它属性
        --------

        x : 状态估计向量 (根据不同阶段可能等于先验估计或者后验估计)

        P : 状态估计误差的协方差矩阵 (即 x 的误差的协方差)

        x_prior : 先验估计状态向量, 即 x 在子空间 { z_i, i<k } 上的最佳线性逼近.

        P_prior : 先验估计误差 (x - x_prior) 的协方差矩阵.

        x_post : 后验估计状态向量, 即 x 在子空间 { z_i, i<=k } 上的最佳线性逼近.

        P_post : 后验估计误差 (x - x_post) 的协方差矩阵.

        Q : 过程噪声协方差矩阵.

        R : 观测噪声协方差矩阵.

        y : 更新一步中的残差向量, 即 z_k 中垂直于 { z_i, i<k } 的分量.

        K : Kalman 增益矩阵.

        z : 最近一次更新中使用的观测向量.

        history : 所有最优估计值的历史记录
        """
        self.dim_x = dim_x
        self.dim_z = dim_z

        self._x = np.zeros(dim_x)
        self._y = np.zeros(dim_z)

        self._x = np.zeros(dim_x)
        if init_x is not None:
            self.set_init_state(init_x)

        self._P = np.eye(dim_x)
        if init_P is not None:
            self.set_init_cov(init_P)

        self._Q = np.eye(dim_x)
        if Q is not None:
            self.set_process_noise_cov(Q)

        self._R = np.eye(dim_z)
        if R is not None:
            self.set_measure_noise_cov(R)

        self._K = np.zeros((dim_x, dim_z))
        self._I = np.eye(dim_x)
        self._z = np.array([None] * self.dim_z)
        self.x_prior = self._x.copy()
        self.P_prior = self._P.copy()

        self.x_post = self._x.copy()
        self.P_post = self._P.copy()

        self.history = []

    def set_init_state(self, x0):
        """
        设置系统初始状态. 必须形如 numpy.array(dim_x).
        """
        x0 = np.asarray(x0)
        if x0.shape != (self.dim_x,):
            raise ValueError("State vector dimension error: {}".format(x0.shape))
        self._x = x0
        self.history.append(x0)

    def set_init_cov(self, P):
        """
        设置初始协方差矩阵. 必须形如 numpy.array(dim_x, dim_x).
        """
        P = np.asarray(P)
        if P.shape != (self.dim_x, self.dim_x):
            raise ValueError("Covariance matrix dimension error: {}".format(P.shape))
        self._P = P

    def set_process_noise_cov(self, Q):
        """
        设置过程噪声协方差矩阵. 必须形如 numpy.array(dim_x, dim_x).
        """
        Q = np.asarray(Q)
        if Q.shape != (self.dim_x, self.dim_x):
            raise ValueError("Covariance matrix dimension error: {}".format(Q.shape))
        self._Q = Q

    def set_measure_noise_cov(self, R):
        """
        设置观测噪声协方差矩阵. 必须形如 numpy.array(dim_z, dim_z).
        """
        R = np.asarray(R)
        if R.shape != (self.dim_x, self.dim_x):
            raise ValueError("Covariance matrix dimension error: {}".format(R.shape))
        self._R = R

    def get_state(self):
        return self._x, self._P

    def predict(self, Fx, JFx, Q=None, f_args=(), jf_args=()):
        """
        预测步骤. 会更新状态为先验估计并更新估计误差的协方差矩阵.

        Fx : 状态转移函数.
        JFx : 一个函数, 对状态 x, 返回 f 在该点处的 Jacobi 矩阵.
        Q : 你可以在这里传入新的过程噪声协方差.
        f_args : fx 函数中使用的其它参数.
        jf_args : jfx 函数中使用的其它参数.
        """
        if not isinstance(f_args, tuple):
            f_args = (f_args,)

        if not isinstance(jf_args, tuple):
            jf_args = (jf_args,)

        if Q is None:
            Q = self._Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_z) * Q

        self._x = Fx(self._x, *f_args)
        JF = JFx(self._x, *jf_args)
        self._P = np.dot(JF, self._P).dot(JF.T) + Q

        self.x_prior = np.copy(self._x)
        self.P_prior = np.copy(self._P)
        return self

    def update(self, z, Hx, JHx, R=None, h_args=(), jh_args=(),
               residual=np.subtract):
        """
        更新步骤. 将先验估计更新为后验估计, 并更新误差的协方差矩阵.

        z : 更新中使用的观测量.
        hx : 观测量的状态转移函数.
        jhx : 一个函数, 对状态 x 返回 h 在该点的 Jacobi 矩阵.
        R : 你可以在这里传入新的观测噪声变量.
        h_args : hx 函数中使用的其它参数.
        jh_args : jhx 函数中使用的其它参数.
        residual : 这是一个函数, 用于计算两个观测量的残差. 默认为 np.subtract.
        """
        # 如果测量值为 None 的话则不作更新,
        # 使用之前的预测值.
        if z is None:
            self._z = np.array([None] * self.dim_z)
            self.x_post = self._x.copy()
            self.P_post = self._P.copy()
            return

        if not isinstance(h_args, tuple):
            h_args = (h_args,)

        if not isinstance(jh_args, tuple):
            jh_args = (jh_args,)

        if R is None:
            R = self._R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        # 计算 Kalman 增益矩阵
        JH = JHx(self._x, *jh_args)
        PHT = np.dot(self._P, JH.T)
        S = np.dot(JH, PHT) + R
        self._K = PHT.dot(np.linalg.inv(S))

        # 计算更新后的估计值
        self._y = residual(z, Hx(self._x, *h_args))
        self._x += np.dot(self._K, self._y)

        # 计算更新后的误差协方差
        I_KH = self._I - np.dot(self._K, JH)
        self._P = np.dot(I_KH, self._P).dot(I_KH.T) + np.dot(self._K, R).dot(self._K.T)
        # 重置 x_post 和 P_post
        self._z = z.copy()
        self.x_post = self._x.copy()
        self.P_post = self._P.copy()

        self.history.append(self._x)
        return self
