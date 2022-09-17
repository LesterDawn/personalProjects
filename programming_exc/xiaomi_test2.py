#!/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re


class ListNode:
    def __init__(self):
        self.data = None
        self.next = None


class Solution:
    def reverseBetween(self, head, left, right):
        newhead = ListNode()
        newhead.data = 0
        newhead.next = head

        pre = newhead
        curr = head
        for i in range(left - 1):  # find left side
            pre = curr
            curr = curr.next
        for i in range(left, right):
            tmp = curr.next
            curr.next = tmp.next
            tmp.next = pre.next
            pre.next = tmp
        return newhead.next


# Write Code Here

head_cnt = int(input())
head = None
head_curr = None
for x in input().split():
    head_temp = ListNode()
    head_temp.data = int(x)
    head_temp.next = None
    if head == None:
        head = head_temp
        head_curr = head
    else:
        head_curr.next = head_temp
        head_curr = head_temp

left = int(input())

right = int(input())

s = Solution()
res = s.reverseBetween(head, left, right)

while res != None:
    print(str(res.data) + " "),
    res = res.next
print("")
