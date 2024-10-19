import numpy as np

a = np.array([[1, 1, 1], [1, 1, 1]])
b = np.array([[1, 1, 0], [0, 1, 1]])

# 检查两个数组的形状是否相同
if a.shape != b.shape:
    raise ValueError("两个数组的形状不相同")

# 找出a中为1但b中不为1的元素
result = (a & ~b).astype(int)

print(result)