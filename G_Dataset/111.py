import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成螺旋数据
theta = np.linspace(0, 4 * np.pi, 100)  # 角度参数
z = np.linspace(0, 1, 100)  # 高度参数
r = z  # 半径随高度线性变化
x = r * np.sin(theta)  # 计算x坐标
y = r * np.cos(theta)  # 计算y坐标

# 生成目标变量 (非线性组合)
# 假设目标变量是 x, y, z 的非线性组合，并加入噪声
noise = np.random.randn(100) * 0.1  # 随机噪声
target = 0.5 * x**2 + 0.3 * y**2 + z + noise  # 非线性目标函数

# 可视化生成的数据
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=target, cmap='viridis', label='Nonlinear data')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

# 生成用于回归的数据集
# X 数据为输入特征 (x, y, z)，target 为输出变量
X = np.column_stack((x, y, z))  # 将 x, y, z 合并为输入特征矩阵

# 输出生成的数据集
import pandas as pd
data = pd.DataFrame(X, columns=['x', 'y', 'z'])
data['target'] = target

# 显示生成的数据集
import ace_tools as tools; tools.display_dataframe_to_user(name="Nonlinear Regression Dataset", dataframe=data)
