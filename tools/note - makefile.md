makefile定义了一系列规则来指定编译规则。make是解释makefile中指令的命令工具，如linux下GNU。默认编译器为UNIX下GCC和CC
# 介绍
例子：GNU的make手册
1）如果这个工程没有编译过，那么我们的所有C文件都要编译并被链接。
2）如果这个工程的某几个C文件被修改，那么我们只编译被修改的C文件，并链接目标程。
3）如果这个工程的头文件被改变了，那么我们需要编译引用了这几个头文件的C文件，并链接目标程序。

## makefile规则
```makefile
target ... : prerequisites ...
command
...
```

- target是目标文件，可以是ObjectFile，也可以是执行文件，还可以是标签

- prerequisites是要生成那个target所需要的文件或者目标

- command是make需要执行的命令（任意shell）

其中包含文件依赖关系，即target目标文件依赖于prerequisites中的文件，生成规则定义在command中，也即prerequisites中若有一个以上文件比target文件要新的话，command定义命令就被执行。

```ma
objects = main.o kbd.o command.o display.o \
insert.o search.o files.o utils.o

edit : $(objects)
cc -o edit $(objects)

main.o : main.c defs.h
cc -c main.c
kbd.o : kbd.c defs.h command.h
cc -c kbd.c
command.o : command.c defs.h command.h
cc -c command.c
display.o : display.c defs.h buffer.h
cc -c display.c
insert.o : insert.c defs.h buffer.h
cc -c insert.c
search.o : search.c defs.h buffer.h
cc -c search.c
files.o : files.c defs.h buffer.h command.h
cc -c files.c
utils.o : utils.c defs.h
cc -c utils.c
clean :
rm edit main.o $(objects)
```

保存为Makefile，输入"make"生成执行文件edit，输入"make clean"删除执行文件和所有中间目标文件。

这个Makefile中，target包括：执行文件edit和中间目标文件(*.o)，依赖文件为冒号后的.c, .h文件，每一个.o文件都有一组依赖文件，这些.o文件又是执行文件edit的依赖文件。依赖文件实质说明了目标文件由哪些文件生成，也即为哪些文件更新。

定义好依赖关系后，后续一行定义了如何生成目标文件的操作系统命令，要以Tab键开头。

这里clean并不是文件，只是一个动作名，冒号后空白，则make不会找文件依赖，不会自动执行后面定义命令。要执行需要make后指出label的名字。

## make如何工作

默认，输入make

1. make会在当前目录下找名字为Makefile或makefile
2. 若找到，会找第一个目标文件target，作为最终目标文件
3. 若target不存在，或者target后面依赖的文件修改时间比target新，则执行后面的命令生成这个target
4. 若target依赖的文件也不存在，则会在当前makefile中寻找所依赖文件的依赖性

所以make会一层层地寻找文件依赖关系，直到编译出第一个目标文件，若出现错误，则直接退出报错。

## make自动推导

GNU的make可以自动推导文件及文件依赖关系后的命令。make看到.o文件，会自动加入.c文件，且cc -c .c也会推导

```makefile
objects = main.o kbd.o command.o display.o \
insert.o search.o files.o utils.o

edit : $(objects)
cc -o edit $(objects)

main.o :  defs.h
kbd.o :  defs.h command.h
command.o : defs.h command.h
display.o : defs.h buffer.h
insert.o : defs.h buffer.h
search.o : defs.h buffer.h
files.o :  defs.h buffer.h command.h
utils.o :  defs.h

.PHONY : clean
clean :
rm edit main.o $(objects)
```

这种方法称为make隐晦规则，.PHONY表示clean是个伪目标文件

另类风格：

```makefile
objects = main.o kbd.o command.o display.o \
insert.o search.o files.o utils.o

edit : $(objects)
cc -o edit $(objects)

$(objects) : defs.h

kbd.o command.o files.o : command.h
display.o insert.o search.o files.o : buffer.h

.PHONY : clean
clean :
rm edit $(objects)
```

每个makefile中都应该写一个清空目标文件

# 总述

## Makefile包含

1. 显示规则：说明如何生成一个或多个目标文件
2. 隐晦规则：make有自动推导功能，可以简略地写Makefile
3. 变量定义：变量一般为字符串
4. 文件指示：在一个Makefile中引用另一个Makefile，类似include；根据某些情况指定Makefile有效部分，类似if；定义一个多行命令
5. 注释：行注释#

命令必须以Tab开始。

## 文件名

默认make寻找"GNUmakefile", "makefile", "Makefile"

## 引用

```makefile
include <filename>
```

filename可以是当前操作系统Shell的文件模式（可以保留路径和通配符）在include前面可以有空字符，但不能是Tab

若有a.mk b.mk c.mk $(bar)包含e.mk f.mk

```makefile
include foo.make *.mk $(bar)
include foo.make a(b c d e).k
```

make会找寻include所有makefile并将内容安置在当前位置，若没有指定绝对或相对路径，make会在当前目录寻找，若没有找到：

1. 执行时，有"-I", "--include -dir"参数，make会在该参数指定目录下寻找
2. 若目录<prefix>/include(usr/local/bin, /usr/include)存在，也会去找。若没有文件找到，make会生成警告命令，但会继续载入，完成后重试，若不可，则出现致命信息，如果想要不理会无法读取的文件继续执行，可使用-，表示无论什么错误都不报错继续执行

```makefile
-include <filname>
```

- 环境变量

若当前环境定义了环境变量MAKEFILES，则make会将这个变量中的值做类似include动作，不过与include不同，这个环境引入的Makefile目标不会起作用，发现错误也不会理会。建议不要使用

## GNU make工作方式

1、读入所有的Makefile。
2、读入被include的其它Makefile。
3、初始化文件中的变量。
4、推导隐晦规则，并分析所有规则。
5、为所有的目标文件创建依赖关系链。
6、根据依赖关系，决定哪些目标要重新生成。
7、执行生成命令。

1-5布为一阶段，一阶段中若定义变量被使用了，make会将其展开在使用的位置。但并不会完全展开，仅当这条依赖被决定要使用，变量才会在内部展开。

# 书写规则

规则包含依赖关系和生成目标的方法，Makefile只有一个最终目标。一般make会以UNIX标准Shell也即/bin/sh

- 通配符

make支持三种*, ?, ...和UNIX的B-Shell相同。~在文件名中用途：~/test表示$HOME路径下的test目录，~hchen/test表示用户hchen宿主目录下的test目录，WINDOWS或MS-DOS下没有宿主目录决定以HOME。

```makefile
targets : prerequisites ; command

objects = *.o
# 变量中通配符
# 目标会依赖于所有文件
objects := $(woldcard *.o)
# 展开，得到所有.o文件集合
```

- 文件搜索

当大量源文件存在不同目录时，加入VPATH使make到指定目录去找。

```makefile
VPATH = src:../headers
```

以上指定两个目录src ../headers，目录由分号分隔，按照顺序搜寻，或使用make的vpath关键字。



1、vpath <pattern> <directories>
为符合模式<pattern>的文件指定搜索目录<directories>。
2、vpath <pattern>
清除符合模式<pattern>的文件的搜索目录。
3、vpath
清除所有已被设置好了的文件搜索目录。
vapth使用方法中的<pattern>需要包含“%”字符。“%”的意思是匹配零或若干字符，例如，“%.h”表示所有以“.h”结尾的文件。<pattern>指定了要搜索的文件集，而<directories>则指定了<pattern>的文件集的搜索的目录。例如：
vpath %.h ../headers
该语句表示，要求make在“../headers”目录下搜索所有以“.h”结尾的文件。（如果某文件在当前目录没有找到的话）

我们可以连续地使用vpath语句，以指定不同搜索策略。如果连续的vpath语句中出现了相同的<pattern>，或是被重复了的<pattern>，那么，make会按照vpath语句的先后顺序来执行搜索。如：
vpath %.c foo
vpath % blish
vpath %.c bar
其表示“.c”结尾的文件，先在“foo”目录，然后是“blish”，最后是“bar”目录。
vpath %.c foo:bar
vpath % blish
而上面的语句则表示“.c”结尾的文件，先在“foo”目录，然后是“bar”目录，最后才是“blish”目录。



- 伪目标，一个标签

```makefile
.PHONY : clean
```

一般没有依赖文件，也可以指定所依赖文件。也可以作为默认目标，只要放在第一个。伪目标也可以成为依赖。

```makefile
all : prog1 prog2 prog3
.PHONY : all
prog1 : prog1.o utils.o
cc -o prog1 prog1.o utils.o
prog2 : prog2.o
cc -o prog2 prog2.o
prog3 : prog3.o sort.o utils.o
cc -o prog3 prog3.o sort.o utils.o
```

以上all伪目标依赖于其他三个目标。由于伪目标总是执行，所依赖的三个目标总会被决议，因此执行多个目标

```makefile
.PHONY: cleanall cleanobj cleandiff
cleanall : cleanobj cleandiff
rm program
cleanobj :
rm *.o
cleandiff :
rm *.diff
```

以上伪目标也可以成为依赖

- 多目标



- 静态模式





# 书写命令





# 使用变量







# 使用条件判断



# 使用函数

# make运行

