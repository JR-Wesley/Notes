---
dateCreated: 2021-10-26
dateModified: 2025-07-27
---
# Basic

1. 方法

```python
#字符串
.upper()
.lower()
.title()#以首字母大写
.rstrip()#删除末尾多余的空格
+#加号连接字符串
.lstrip()#开头
.strip()#两端
# 以上均为临时，对某个对象使用
```

1. 数

```python
str()# 函数，让非字符表示为字符
```

1. 列表

```python
a=['f',1]
print(str(a[-1]))
a.append(3.2)#末尾插入
a.insert(0,'f')#索引插入元素
delete a[2]#删除索引
b=a.pop(x)#提取弹出索引x的元素
a.move('f')#直接删除对应值
```

组织列表

```python
a.sort()
a.sort(reverse=True)#首字母顺序，反序，且为永久性
sorted(a)#临时性函数
a.reverse()#永久翻转顺序
len(a)#获得长度rww
```

1. 操作列表

```python
#循环列表
for n in a:
    do()
    #循环依靠缩进判断结构
range(1,n)#1,2...,n-1
range(1,n,l)#步长l
list(range())#生成列表
square = [value**2 for value in range(1, 11, 2)]#列表解析创建数值
#对数字列表的函数
max()
min()
sum()
#切片提取列表
a[0:-2]
#需要用切片复制，而不是赋值
```

1. 元组：不可变的列表

```python
d = (100, 20)
d = (200, 'a')
#可以重新给变量赋值，而不能改变
```

1. if

```python
if(a.lower() == b and a.upper() != c):
    do()#判断语句
elif  :
else  :
#使用and,or多个判断
#in, not in判断是否包含
if 列表
#判断列表是否为空
```

1. 字典

```python
a = {'color': 'green', 'points': 5}
#一个键对应一个值
a['color']
#通过键访问值，添加，修改，删除
del a['points']
#遍历，通过items()遍历，使用.keys()遍历键，使用.values()访问值
for key,value in a.items()
sorted()#函数对键排序
set()#函数，剔除重复的元素
#列表、字典也可以相互嵌套
```

1. 用户输入和 while

```python
in = input();
i = int();
#可设置sign标志，在while中修改。
#用sign标志来判断退出
while statement:
    do()#
    break#使用break可以跳出任何循环
    continue#使用continue跳过剩余语句
```

1. 函数

```python
def functionname(username,name='dog',choose=' '):
    #将可选的值给默认值空白放在末尾
    do()
    #define关键字表示函数定义
    """文档字符串的注释"""
    #所有缩进构成函数体
    return 
	#返回值，可返回包括字符、列表、字典等数据结构
    
functionname(name= ,username=)
#调用
#传递实参，位置实参或者关键值实参，指定默认值

#禁止函数修改列表，使用切片传递副本，但应该避免复制
function_name(list_name[:])
#传递任意数量的实参
def function_name(names_1，*names):
    #前者为位置实参，后者创建名为names的空元组，封装接收的所有值
def build_profile(first,last,**user_info):
    profile={};
    profile['first_name']=first
    profile['last_name']=last
    for key,value in user_info.items():
        profile[key]=value
    return profile
    #创建任意数量的关键字实参，names为字典，接收任意数量的键-值对
    #**创建字典
user_profile=build_profile('a','e',location='p',field='physics')
```

1. 类

```python
class Car():
    """首字母大写为类名"""
    #类中的函数称为方法。
    def __init__(self,name, year):
    """创建对象时自动运行，且自动传入实参self，形参self必须位于前面，每个与类相关的方法都将传入实参self，是一个指向实例本身的引用，让实例能够访问类中的属性和方法，只需要给后面的形参提供值"""
    #以self为前缀的变量都可供类中的所有方法使用，还可以通过类的任何实例来访问这些变量。像这样可通过实例访问的变量称为属性。
    self.agu=0
    #有初始值的属性可以不包含形参
    #通过句点访问，方法改变属性的值
    def fill():
        
       	
    #继承
class Elecar(car):
    def __init__(self, name, year):
        super.__init__(name, year)
    #super将父类和子类联系起来
    def fill():
    #可覆盖父类的同名方法。
    
#从模块导入类，导入外部模块，标准库
```

1. 文件和异常

```python
with open('pi_digits.txt') as file_object:
	contents = file_object.read()
	print(contents)
    #with在不需要访问时关闭文件，不需要额外close()
    #read()方法读取，作为字符串存储，末尾多出一空行，int(),float()可以转换
    #Windows中\寻址，linux和OSX中/地址
    for line in file_object:
        #逐行读取
    lines = file_object.readlines()
    	#创建包含各行的列表
    with open(filename, 'w') as file_object:
        file_object.write("I love programming.")
    	#写入文件，打开文件时，可指定读取模式（'r'）、写入模式（'w'）、附加模式（'a'）或让读取和写入文件的模式（'r+'）。省略了模式实参，默认只读模式。不存在则创建，不会添加换行。
```

```python
try:
	print(5/0)
except ZeroDivisionError:
	print("You can't divide by zero!")
else:
	print(answer)
#如果异常，避免崩溃，成功则继续运行
#方法split()以空格为分隔符将字符串分拆成多个部分，并将这些部分都存储到一个列表中，结果是一个包含字符串中所有单词的列表，虽然有些单词可能包含标点。
```

1. 存储数据

```python
#使用json模块存储数据
import json

with open(filename, 'w') as f_obj:
    json.dump(nubers, f_obj)
    #接受两个参数，数据和文件
    numbers=json.load(f_obj)
```

1. 测试代码

```python
#unittest提供了代码测试工具，
import unittest
from name_function import get_formatted_name
class NamesTestCase(unittest.TestCase):
    def test_name(self):
        #只包含一个方法，用于测试函数的一个方面，以test_开头的方法会自动运行。
        self.assertEqual(formmatted_name,'**')
        #unittest功能——断言方法，
assertEqual(a, b) 核实a == b
assertNotEqual(a, b) 核实a != b
assertTrue(x) 核实x为True
assertFalse(x) 核实x为False
assertIn(item, list) 核实item在list中
assertNotIn(item, list) 核实item不在list中

def setUp():
    #方法，创建队形，供测试使用
```

1. print

```python
print(*objects, sep=' ', end='\n', file=sys.stdout)
"""
objects --表示输出的对象。输出多个对象时，需要用 , （逗号）分隔。
sep -- 用来间隔多个对象。
end -- 用来设定以什么结尾。默认值是换行符 \n，我们可以换成其他字符。
file -- 要写入的文件对象。
"""

print(a,"bc""d")
#大部分变量都可以直接输出，','会输出空格，输出默认换行
print('\n')#两行
print()#一行
print(x,end='')#end可以为' ',','

print('abx,%s,%d,%10.3f'%(s,d,PI))
#和C类似，%格式控制和转换说明分格
print('PI=%.*f'%(3,PI))#精度
print('PI=%*.f'%(3,PI))#长度
#%后可以接+-，space，0表补充

```

1. import

```python
# 将函数存储在模块中，模块是.py文件
mod_n.py
def func_n():

#导入模块
import mod_n
mod_n.func_n()
#导入特定函数
from mod_n import func_n,func_n_2
func_n()#不需要使用句点

import mod_n as nn
#关键字重命名函数，模块
from mod_n import *
#导入所有函数
```

# 常用模块使用

## Re 正则表达式

https://www.runoob.com/python/python-reg-expressions.html

## FIle 文件读写

https://www.runoob.com/python/file-methods.html

## Collections 模块之 Counter()

# 高级特性

https://huccihuang.github.io/

https://python-cookbook.readthedocs.io/zh-cn/latest/

fluent python
