import numpy as np

# 构造数据
n_samples = 100
n_features = 20

X = np.random.rand(n_samples, 2, n_features)  # 创建100x2x20的随机矩阵，作为自变量，也就是我们说的Y
beta_true = np.random.rand(n_features, 1)  # 真实的20x1系数矩阵，也就是力矩
y_true = np.matmul(X, beta_true)  # 我们要求的pai

# 将X和y重塑为适合np.linalg.lstsq的形状
X_reshaped = X.reshape(-1, n_features)
y_reshaped = y_true.reshape(-1, 1)

# 使用NumPy的最小二乘法函数
beta_estimate, residuals, rank, s = np.linalg.lstsq(X_reshaped, y_reshaped, rcond=None)

print("True coefficients:", beta_true)
print("Estimated coefficients:", beta_estimate)
