#!/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re


# 请完成下面这个函数，实现题目要求的功能
# 当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# ******************************开始写代码******************************

class BiTreeNode:
    def __init__(self, data):
        self.data = data
        self.lchild = None
        self.rchild = None


def solution(input):
    if not input or len(input) == 1:
        return input
    root = input[0]
    for i, s in enumerate(input):



# ******************************结束写代码******************************


try:
    _input = input()
except:
    _input = None

res = solution(_input)

print(res + "\n")