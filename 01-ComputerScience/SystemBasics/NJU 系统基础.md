
- 该目录整理了 MOOC 计算机系统基础课

```dataview
LIST "Computer System Basics"
FROM ""
WHERE file.folder = this.file.folder OR startswith(file.folder, this.file.folder + "/")
SORT file.path
```
