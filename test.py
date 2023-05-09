from functools import reduce

# 定义一个列表
lst = [1, 2, 3, 4, 5]

# 使用reduce()函数对列表中的元素求和
result = reduce(lambda x, y: x**y, lst)
print(result)  # 输出15（即1+2+3+4+5）









