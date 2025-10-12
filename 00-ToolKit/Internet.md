---
dateCreated: 2025-08-10
dateModified: 2025-08-10
tags:
  - Tool
---

# Java

[如何Linux环境下安装最新JDK19？一篇文章带你详细了解！-云社区-华为云 (huaweicloud.com)](https://bbs.huaweicloud.com/blogs/386171)

[Ubuntu：配置环境变量的两种常用方法（ .bashrc 和 /etc/profile ）_ubuntu bashrc_微步_ym的博客-CSDN博客](https://blog.csdn.net/yiminghd2861/article/details/98854882)

## Sbt

对于 sbt 的下载，有很多教程直接 sudo apt install sbt，有的还在之前先 update 一下，但这个方法至少对我来说一直没有用。最保险的方法是上官网下载，可以试试官网提供的命令行下载方法（没亲测过）:

也可以直接下载. tgz，解压后同安装 scala，vim ~/. bashrc 在最后添加：

```
export SBT_HOME=安装路径/sbt
export PATH=$SBT_HOME/bin:$PATH
```

最后 source ~/. bashrc 更新。

测试 sbt，在任意文件夹下输入 sbt sbtVersion，若出现版本号说明安装成功。

可以对 sbt 换源，进入~/. sbt，创建文件 repositories（？）

# Kaggle

https://zhuanlan.zhihu.com/p/693008163

```json
{
	"request": [
		{
			"enable": true,
			"name": "Google APIs",
			"ruleType": "redirect",
			"matchType": "regexp",
			"pattern": "^http(s?)://ajax\\.googleapis\\.com/(.*)",
			"exclude": "",
			"isFunction": false,
			"action": "redirect",
			"to": "https://gapis.geekzu.org/ajax/$2",
			"group": "Google Redirect"
		},
		{
			"enable": true,
			"name": "reCaptcha",
			"ruleType": "redirect",
			"matchType": "regexp",
			"pattern": "^http(s?)://(?:www\\.|recaptcha\\.|)google\\.com/recaptcha/(.*)",
			"exclude": "",
			"isFunction": false,
			"action": "redirect",
			"to": "https://recaptcha.net/recaptcha/$2",
			"group": "Google Redirect"
		}
	],
	"sendHeader": [],
	"receiveHeader": [
		{
			"enable": true,
			"name": "Content Security Policy Header Modification",
			"ruleType": "modifyReceiveHeader",
			"matchType": "all",
			"pattern": "",
			"exclude": "",
			"isFunction": true,
			"code": "let rt = detail.type;\nif (rt === 'script' || rt === 'stylesheet' || rt === 'main_frame' || rt === 'sub_frame') {\n  for (let i in val) {\n    if (val[i].name.toLowerCase() === 'content-security-policy') {\n      let s = val[i].value;\n      s = s.replace(/googleapis\\.com/g, '$& https://gapis.geekzu.org');\n      s = s.replace(/recaptcha\\.google\\.com/g, '$& https://recaptcha.net');\n      s = s.replace(/google\\.com/g, '$& https://recaptcha.net');\n      s = s.replace(/gstatic\\.com/g, '$& https://*.gstatic.cn');\n      val[i].value = s;\n    }\n  }\n}",
			"group": "Google Redirect"
		}
	]
}
```

qv2ray

https://zhuanlan.zhihu.com/p/30761252365

# 云服务器

要理解**云服务器**、**云主机**和**裸金属服务器**的区别，核心是从“资源隔离方式”“硬件占用形式”“适用场景”三个维度切入——三者本质是云计算架构中不同层级的计算资源交付形态，对应不同的性能、灵活性和成本需求。


### 一、先明确：“云服务器”与“云主机”的关系（常被混淆）
在国内云计算语境中，**“云服务器”和“云主机”几乎是同义词**，只是不同厂商的命名习惯差异：
- 阿里云、腾讯云等厂商多称“云服务器”（如ECS、CVM）；
- 部分厂商或早期文档会称“云主机”，本质都是基于**虚拟化技术**的“共享硬件资源的虚拟计算实例”。

简单说：**云主机 = 云服务器**（下文统一用“云服务器”代称，避免重复），二者与“裸金属服务器”是明确的对立概念。


### 二、核心区别：云服务器 vs 裸金属服务器
云服务器（虚拟化）和裸金属服务器（物理机）的本质差异，是“是否通过虚拟化层隔离资源”，这直接决定了它们的性能、灵活性和适用场景。以下是详细对比：

| 对比维度                | 云服务器（虚拟化）                                                                 | 裸金属服务器（物理机）                                                               |
|-------------------------|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| **硬件占用形式**        | 共享物理硬件：多台云服务器通过“虚拟化层”（如KVM、VMware）共享一台物理机的CPU、内存、硬盘资源。 | 独占物理硬件：一台裸金属服务器对应一台完整的物理机，CPU、内存、硬盘等硬件资源100%归用户独占。 |
| **资源隔离性**          | 逻辑隔离：通过虚拟化技术划分资源，但底层硬件仍共享，极端场景下可能存在“资源争抢”（如CPU调度延迟）。 | 物理隔离：硬件完全独立，无资源争抢，安全性和稳定性远高于云服务器。                     |
| **性能表现**            | 性能有“损耗”：虚拟化层会占用部分硬件资源（约5%-10%），不适合对CPU/内存敏感的高负载场景。 | 性能无损耗：直接调用物理硬件，支持CPU超线程、PCIe直通（如GPU、SSD），性能等同于物理机。 |
| **灵活性与弹性**        | 弹性极强：支持“秒级创建/销毁”“按需扩容”（如CPU从2核扩至32核，内存从4G扩至128G），按使用时长计费（小时/天）。 | 弹性较弱：硬件资源固定，扩容需更换物理机（耗时数小时至数天），多按“月/年”长期计费。   |
| **管理复杂度**          | 低：厂商提供可视化控制台，无需关注硬件维护（如硬件故障由厂商自动迁移数据、更换硬件），用户仅需管理操作系统和应用。 | 高：需自行管理硬件（如驱动安装、硬件故障排查），部分厂商提供“托管裸金属”（代维护硬件），但仍需关注底层配置。 |
| **成本**                | 低门槛、按需付费：起步价低（如1核2G配置每月几十元），适合中小负载。                 | 高成本：单台价格数千元/月起，适合长期、高负载场景，成本接近自购物理机但省去机房维护费。 |
| **典型适用场景**        | 1. 中小型网站/博客、小程序后端；<br>2. 开发测试环境（临时创建，用完销毁）；<br>3. 轻量应用（如CRM、OA系统）。 | 1. 高性能计算（如AI训练、大数据分析）；<br>2. 核心业务数据库（如MySQL、Oracle集群）；<br>3. 对稳定性要求极高的场景（如金融交易、工业控制）。 |


### 三、延伸：三者与“传统物理机”的关系
很多人会混淆“裸金属服务器”和“传统物理机”，其实二者的核心差异是“交付方式”：
- **传统物理机**：需自行采购硬件、搭建机房、部署网络，维护成本高，交付周期长（数周）；
- **裸金属服务器**：由云厂商提供硬件和机房，用户通过云控制台“秒级开通”，硬件维护由厂商负责，本质是“云化的物理机”。

而云服务器（虚拟化）是“传统物理机的虚拟化切片”，三者的技术演进路径是：  
**传统物理机 → 云服务器（虚拟化，提升资源利用率） → 裸金属服务器（云化物理机，兼顾性能与云管理）**


### 四、总结：如何选择？
1. **选云服务器（虚拟化）**：如果你的需求是“灵活、低成本、轻负载”，或需要临时环境（如开发测试），优先选云服务器；  
2. **选裸金属服务器**：如果你的需求是“高性能、高稳定、无资源争抢”（如数据库、AI训练），且能接受较高成本，优先选裸金属；  
3. 无需纠结“云主机”：遇到“云主机”概念时，直接等同于“云服务器”即可，无需额外区分。