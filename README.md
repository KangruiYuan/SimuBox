
# *SimuBox* for science calculation

[![Static Badge](https://img.shields.io/badge/GitHub-SimuBox-7C8EFF?logo=github)](https://github.com/KangruiYuan/SimuBox.git)
[![Static Badge](https://img.shields.io/badge/PyPI-SimuBox-B39CD0?logo=pypi)](https://pypi.org/project/SimuBox/)
![Static Badge](https://img.shields.io/badge/python-version_3.8%2B-blue?logo=python&logoColor=white)
[![Static Badge](https://img.shields.io/badge/DOI-doi.org%2F10.1002%2Fchem.202301043-purple)](https://doi.org/10.1002/chem.202301043)


## 愿景

> 生活的全部意义在于无穷地探索尚未知道的东西，在于不断地增加更多的知识。 —— 左拉 《萌芽》

该仓库实现了一些科学计算，尽管大部分是为了实现自己的需求，功能也相对驳杂，但依旧希望经过整理后能够帮到部分人。

## 版本管理

以下的版本号仅作相对参考。

`GitHub`仓库版本: ***0.1.11***

`PyPI` 版本: ***0.1.6***

## 联系方式

该库为本人求学期间为完成科研工作而常用的模块的集合，可能并不完善。任何问题请发布在`issue`或者通过邮件联系本人。
`E-mail：kryuan@qq.com`

# _安装引导_

- 方法一：通过git下载：

`git clone https://github.com/KangruiYuan/SimuBox.git`

最后将其配置到python环境中，如在`site-packages`目录下创建`.pth`文件，并在该文件中写入路径。

- 方法二：通过pip下载

`pip install SimuBox`

## `Requirements` 环境要求

```
python=3.9
pandas >= 1.4
scipy >= 1.10.0
matplotlib >= 3.6.2
numpy >= 1.19.5
opencv-contrib-python >= 4.7.0.72
opencv-python >= 4.5.3.56
```
其他第三方库，若无冲突可以直接使用最新版。

# 使用示例

示例请见[`/demo/Local/base.ipynb`](https://github.com/KangruiYuan/SimuBox/blob/main/demo/Local/base.ipynb)