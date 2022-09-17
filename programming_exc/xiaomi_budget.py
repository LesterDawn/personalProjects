#!/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re


# 请完成下面这个函数，实现题目要求的功能
# 当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# ******************************开始写代码******************************


def solution(prices, budget):
    prices.sort()
    item_num = 0
    for i in range(len(prices) - 1, -1, -1):
        item_num += int(budget / prices[i])
        budget = budget % prices[i]
        if budget == 0:
            break
        if min(prices) > budget:
            item_num = -1
            break
    return item_num


# ******************************结束写代码******************************


_prices_cnt = 0
_prices_cnt = int(input())
_prices_i = 0
_prices = []
while _prices_i < _prices_cnt:
    _prices_item = int(input())
    _prices.append(_prices_item)
    _prices_i += 1

_budget = int(input())

res = solution(_prices, _budget)

print(str(res) + "\n")