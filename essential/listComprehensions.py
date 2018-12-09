# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 11:58:50 2018

@author: alper
"""

"""
new_list = []
for i in old_list:
    if filter(i):
        new_list.append(expressions(i))
        
"""

"""
new_list = [expression(i) for i in old_list if filter(i)]
"""

"""
The list comprehension starts with a '[' and ']', to help you remember that the
result is going to be a list.

The basic syntax is
"""

"""
[ expression for item in list if conditional ]

This is equivalent to:
    
for item in list:
    if conditional:
        expression
"""

x = [i for i in range(15)]
print (x)

squares = [x**2 for x in range(10)]
print(squares)

squares = []
for i in range(10):
    print(squares)
    squares.append(i**2)

upper = [x.upper() for x in ["a","b","c"]]
print(upper)

lower = [x.lower() for x in ["ALPEREN","HALIL","OTHERS"]]
print(lower)

string = "Hello 232423 Krakow with 609324023" 
numbers = [x for x in string if x.isdigit()]
print(numbers)

string = "Hello 232423 Krakow with 609324023" 
numbers = [x for x in string if x.isalpha()]
print(numbers)

fh = open("text.txt",'r')
result = [i for i in fh if "line3" in i]
print(result)

def double(x):
    return x**2

even = [double(x) for x in range(25) if x%2==0]
print(even)