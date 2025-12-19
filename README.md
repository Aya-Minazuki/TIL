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
