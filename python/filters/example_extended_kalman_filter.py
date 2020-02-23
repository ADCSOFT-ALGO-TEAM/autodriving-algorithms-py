"""
扩展 Kalman 滤波示例代码
"""
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from EKF import ExtendedKalmanFilter


# 时间间隔
dt = 0.1
# 总时间
total_time = 60


def get_control_input(x):
    """
    给定状态 x, 返回对应的控制向量. 这里假定控制速度为常数, 航向角均匀变化.
    """
    v = 1.0
    yawrate = 0.1
    u = np.array([v, yawrate])
    return u


def Fx(x, u):
    """
    给定状态 x 和控制 u, 返回 f(x, u) 的值.
    """
    yaw = x[2]
    F = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]])

    B = np.array([[dt * np.cos(yaw), 0],
                  [dt * np.sin(yaw), 0],
                  [0, dt],
                  [1, 0]])

    return np.dot(F, x) + np.dot(B, u)


def JFx(x, u):
    """
    给定状态 x 和控制 u, 返回 f(x, u) 在该点处的 Jacobi 矩阵.

    车辆运动模型:

        x_{t+1} = x_t + v * dt * cos(yaw)
        y_{t+1} = y_t + v * dt * sin(yaw)
        yaw_{t+1} = yaw_t + omega * dt
        v_{t+1} = v_t

    所以
        dx / d(yaw) = -v * dt * sin(yaw)
        dy / d(yaw) =  v * dt * cos(yaw)
        dx / dv     = dt * cos(yaw)
        dy / dv     = dt * sin(yaw)
    """
    yaw = x[2]
    v = u[0]
    return np.array([[1, 0, -dt*v*sin(yaw), dt*cos(yaw)],
                     [0, 1,  dt*v*cos(yaw), dt*sin(yaw)],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def Hx(x):
    """
    给定状态 x, 返回 h(x) 的值.
    """
    z = x[:2]
    return z


def JHx(x):
    """
    给定状态 x, 返回 h(x) 在 x 处的 Jacobi 矩阵.
    """
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])


def add_gps_noise(x):
    gps_noise = np.diag([0.5, 0.5]) ** 2
    return Hx(x) + np.dot(gps_noise, np.random.randn(2))


def add_input_noise(u):
    input_noise = np.diag([1.0, np.deg2rad(30.0)]) ** 2
    return u + np.dot(input_noise, np.random.randn(2))


def plot_covariance_ellipse(x, P, *args, **kwargs):
    """
    绘制二维协方差矩阵对应的椭圆.

    x : 车辆状态向量.
    P : 估计误差的协方差矩阵.
    """
    Pxy = P[:2, :2]
    # 协方差矩阵的特征值对应椭圆的长短轴长度的平方,
    # 特征向量对应椭圆长短轴的方向.
    eigval, eigvec = np.linalg.eigh(Pxy)
    # 确定长短轴的下标
    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    # 在椭圆上按照角度均匀采点
    t = np.arange(0, 2 * np.pi + 0.1, 0.1)
    # 长短轴长度
    a = np.sqrt(eigval[bigind])
    b = np.sqrt(eigval[smallind])
    pt = (a * np.cos(t), b * np.sin(t))
    # 长轴倾斜角度
    angle = np.arctan2(eigvec[bigind, 1], eigvec[bigind, 0])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    # 将椭圆上的点按照长轴的倾斜角度进行旋转
    fx = np.dot(R, pt)
    # 然后平移到坐标 (x, y) 处
    px = fx[0] + x[0]
    py = fx[1] + x[1]
    plt.plot(px, py, *args, **kwargs)


def display_trajectory(data, *args, **kwargs):
    xlist, ylist = np.asarray(data).T[:2]
    plt.plot(xlist, ylist, *args, **kwargs)


def main():
    ekf = ExtendedKalmanFilter(dim_x=4,
                               dim_z=2,
                               Q=np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2,
                               R = np.diag([1.0, 1.0]) ** 2,
                               init_x=np.zeros(4),
                               init_P=np.eye(4))
    x0, _ = ekf.get_state()
    x_true = x_dr = x0
    x_true_list = [x0]   # "真实' 理论值历史记录 (无观测, 不计任何噪声)
    x_dr_list = [x0]     # 航迹推算的理论值历史记录 (无观测, 计入过程噪声)
    z_list = []          # 观测值列表历史记录
                         # 估计值的历史记录 (有观测, 计入过程和观测噪声) 在 ekf 内部有记录

    current_time = 0
    while current_time <= total_time:
        plt.cla()
        # 第一个人只相信理论模型，完全不信观测值, 而且不愿意考虑过程噪声 (真实理论值 x_true)
        u = get_control_input(x_true)
        x_true = Fx(x_true, u)
        x_true_list.append(x_true)
        display_trajectory(x_true_list, "b-", label="true model")

        # 第二个人只相信理论模型, 完全不信观测值, 但是愿意考虑过程噪声 (航迹推算)
        u_dr = get_control_input(x_dr)
        u_dr = add_input_noise(u)
        x_dr = Fx(x_dr, u_dr)
        x_dr_list.append(x_dr)
        display_trajectory(x_dr_list, "k-", label="dead reckoning")

        # 第三个人只相信观测值, 完全不相信理论值
        z = add_gps_noise(x_true)
        z_list.append(z)
        display_trajectory(z_list, "g.", label="measurements")

        # 第四个人二者都不信, 但二者都考虑, 用扩展 Kalman 滤波作理论值和观测值的融合处理
        x_est = ekf.get_state()
        u_est = get_control_input(x_est)
        u_est = add_input_noise(u_est)
        ekf.predict(Fx, JFx, f_args=u_est, jf_args=u_est)
        ekf.update(z, Hx, JHx)
        display_trajectory(ekf.history, "r-", label="ekf estimation")
        x, P = ekf.get_state()
        plot_covariance_ellipse(x, P, "r--", label="ekf covariance ellipse")

        plt.axis("equal")
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.pause(0.001)

        current_time += dt


if __name__ == "__main__":
    main()
