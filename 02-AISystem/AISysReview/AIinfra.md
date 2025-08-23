
```dataview
LIST "AI Infra"
FROM ""
WHERE file.folder = this.file.folder OR startswith(file.folder, this.file.folder + "/")
SORT file.path
```

https://www.infoq.cn/article/edwy1v3xy14pgkefdv1u

三大缩放定律： https://www.chaspark.com/#/hotspots/1174432473590185984
