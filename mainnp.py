import numpy as np

def svd_decomposition(matrix):
    # 计算矩阵的共轭转置
    matrix_h = matrix.conj().T

    # 计算矩阵与其共轭转置的乘积
    matrix_matrix_h = matrix @ matrix_h

    # 计算特征值和特征向量
    eigenvalues_matrix, eigenvectors_matrix = np.linalg.eigh(matrix_matrix_h)

    # 对特征值进行降序排列，并取绝对值
    abs_eigenvalues_matrix = np.abs(eigenvalues_matrix)
    sorted_indices_matrix = abs_eigenvalues_matrix.argsort()[::-1]

    # 对特征向量进行相应的排序
    eigenvectors_matrix = eigenvectors_matrix[:, sorted_indices_matrix]

    # 计算奇异值
    singular_values = np.sqrt(abs_eigenvalues_matrix[sorted_indices_matrix])

    # 构建 Sigma 矩阵
    Sigma = np.diag(singular_values)

    # 构建 U 矩阵
    U = eigenvectors_matrix

    # 构建 V 矩阵
    V = np.linalg.inv(Sigma) @ np.linalg.inv(U) @ matrix

    return U, Sigma, V


# 示例复数矩阵
matrix = np.array([[1.0 + 2.0j, 3.0 - 1.0j],
                   [4.0 + 1.0j, 2.0 + 3.0j]])

# 调用 SVD 分解函数
U, Sigma, V_h = svd_decomposition(matrix)

# 验证 U 和 V 是否满足幺正条件
test1 = U @ U.conj().T
test2 = V_h @ V_h.conj().T

# 验证奇异值是否非负
is_singular_values_nonnegative = np.all(Sigma >= 0)

# 打印结果
print("\n原始矩阵:")
print(matrix)
print("\n自实现的 SVD 分解结果:")
print("U 矩阵:")
print(U)
print("Sigma 矩阵:")
print(Sigma)
print("V 的共轭转置矩阵:")
print(V_h)
print("U@S@V_hermitian")
print(U @ Sigma @ V_h)

# 打印验证结果
print("U 是否满足幺正条件:\n", np.allclose(test1, np.eye(test1.shape[0])))
print("V 是否满足幺正条件:\n", np.allclose(test2, np.eye(test2.shape[0])))
print("奇异值是否非负:", is_singular_values_nonnegative)