这是我的进行人工智能学习的过程
我将每天进行更新我的学习内容，希望激励到自己，并对自己的学习起到帮助。
Python学习
1768. 交替合并字符串
给你两个字符串 word1 和 word2 。请你从 word1 开始，通过交替添加字母来合并字符串。如果一个字符串比另一个字符串长，就将多出来的字母追加到合并后字符串的末尾。
返回 合并后的字符串 。
class Solution:
    def mergeAlternately(self,word1: str, word2: str) -> str:
        result = []
        i, j = 0, 0

        while i < len(word1) and j < len(word2):
            result.append(word1[i])
            i += 1
            result.append(word2[j])
            j += 1

        if i < len(word1):
            result.append(word1[i:])
        if j < len(word2):
            result.append(word2[j:])

        return ''.join(result)
''.join是操作字符串的方式，意思是用单引号内的内容把字符串串联起来
为什么用 word1[i:]（冒号的作用）
Python 中 字符串[起始索引:] 是切片语法，表示：
从 起始索引 开始，取到字符串的最后一个字符（包含起始索引）。
如果 起始索引 超出字符串长度，返回空字符串（不会报错）。

389. 找不同
给定两个字符串 s 和 t ，它们只包含小写字母。
字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。
请找出在 t 中被添加的字母。

class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        count = [0] * 26  # 26个小写字母

    # 统计s中的字母
        for ch in s:
            count[ord(ch) - ord('a')] += 1

    # 遍历t，减去计数
        for ch in t:
            index = ord(ch) - ord('a')
            count[index] -= 1
            if count[index] < 0:
                return ch
        return ""  # 理论上不会走到这里


import numpy as np  # 行业惯例，大家都叫它 np

# 1. 把普通列表变成 Numpy 数组
py_list = [1, 2, 3]
np_array = np.array([1, 2, 3])

print(np_array)
# 输出: [1 2 3]  (注意：它打印出来没有逗号分隔，这是它的小特征)

Python基础循环

data = [1, 2, 3]
new_data = []
for x in data:
    new_data.append(x + 1)
# 结果: [2, 3, 4]

numpy消灭循环

data = np.array([1, 2, 3])
# 见证奇迹的时刻：直接对整个数组加 1
new_data = data + 1 
# 结果: array([2, 3, 4])

在 AI 报错里，90% 的错误都是 Shape Mismatch（形状不匹配）。 你必须学会像呼吸一样自然地查看数据的“形状”。

1D (向量): [1, 2, 3] -> Shape: (3,)

**2D (矩阵/黑白图):**Excel 表格一样 -> Shape: (行数, 列数)


3D (张量/彩色图): 立方体 -> Shape: (高度, 宽度, 颜色通道)

# 创建一个 2行 3列 的矩阵
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(matrix.shape)
# 输出: (2, 3) 
# 意思是：我有 2 个行，每行有 3 个数。

为什么这很重要？ 当你做 AI 项目时，拿到一个数据，第一件事永远不是看它里面的数字是几，而是敲一行 print(data.shape)。

如果输出 (60000, 28, 28)，老手一眼就知道：这是 6 万张 28x28 像素的图片。

如果输出 (2,)，那可能只是个二维坐标点。

做 AI 时，我们经常需要凭空变出一些数据来（比如初始化神经网络的参数权重）。Numpy 提供了“造物主”指令。

# 1. 造一个全 0 的矩阵（通常用于占位）
# 造一个 3行4列 的全零矩阵
zeros = np.zeros((3, 4)) 

# 2. 造一个全 1 的矩阵
ones = np.ones((3, 4))

# 3. 造一个随机矩阵（AI 里的权重初始化核心）
# 生成 3行4列 的随机数（服从标准正态分布）
weights = np.random.randn(3, 4)

(带小数点的)。

原因： np.zeros 默认生成的是 浮点数 (float64)。

AI 里的坑： 在深度学习里，神经网络的权重通常需要 float32（为了跑得快且省显存），而真实的图片文件通常是 uint8（0-255 的整数）。

怎么改？ 如果你想指定它就是整数，可以在创建时加个参数：
fake_image = np.zeros((5, 5), dtype=int)

以后你写 PyTorch 的 Dataset 代码时，经常需要处理这种 dtype 转换，否则模型会报错。
