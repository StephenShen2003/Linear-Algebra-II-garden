import torch
torch.set_default_dtype(torch.float64)


def frobenius(A):
    return torch.sqrt(torch.sum(torch.abs(A)**2))


def generate_random_vector(n):
    real_part = torch.floor(torch.rand(n, 1) * 10)
    imag_part = torch.floor(torch.rand(n, 1) * 10)
    vector = real_part + imag_part * 1j
    return vector


def normalize_vector(vector):
    norm = frobenius(vector)
    if norm == 0:
        return vector
    else:
        normalized_vector = vector / norm
        return normalized_vector


def is_unitary(matrix):
    conjugate_transpose = matrix.conj().t()
    product = matrix @ conjugate_transpose
    return product


def efunc(a, b):
    inner_product = a.conj().t() @ b
    norm_inner_product = frobenius(inner_product)
    if norm_inner_product == 0:
        return 0
    else:
        ne = inner_product[0, 0] / norm_inner_product
        return ne


def wfunc(a, b):
    numerator = efunc(a, b) * a - b
    denominator = frobenius(numerator)
    if denominator == 0:
        return torch.zeros_like(a)
    else:
        w = numerator / denominator
        return w


def householder(a, b):
    h = efunc(a, b) * (torch.eye(n) - 2 * (wfunc(a, b) @ wfunc(a, b).conj().t()))
    return h


def givens(a, b):
    mid = a + b
    mid = normalize_vector(mid)
    print(householder(a,mid))
    print(householder(mid,b))
    g = householder(mid, b)@householder(a, mid)
    return g


# 设置维度和验证次数
n = 2
num_tests = 3

for i in range(num_tests):
    print(f"case {i + 1}:")

    # 随机生成非零向量𝝃和𝜼
    xi = generate_random_vector(n)
    eta = generate_random_vector(n)

    # 将𝜼归一化,并调整𝝃的模长与𝜼相等
    eta = normalize_vector(eta)
    xi = normalize_vector(xi)

    h = householder(xi, eta)
    g = givens(xi, eta)
    hxi = h @ xi
    gxi = g @ xi

    print("Original 𝝃:")
    print(xi)
    print("Original 𝜼:")
    print(eta)
    print("Householder matrix U:")
    print(h)
    print("U𝝃-𝜼:")
    print(hxi - eta)
    print("UU^t-I")
    print(is_unitary(h) - torch.eye(n))
    print("Givens matrix U:")
    print(g)
    print("U𝝃-𝜼:")
    print(gxi - eta)
    print("UU^t-I")
    print(is_unitary(g) - torch.eye(n))

    print()