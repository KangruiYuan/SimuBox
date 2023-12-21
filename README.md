
# *SimuBox* for science calculation

[![Static Badge](https://img.shields.io/badge/GitHub-SimuBox-7C8EFF?logo=github)](https://github.com/KangruiYuan/SimuBox.git)
[![Static Badge](https://img.shields.io/badge/PyPI-SimuBox-B39CD0?logo=pypi)](https://pypi.org/project/SimuBox/)
![Static Badge](https://img.shields.io/badge/version%3E3.9.7-white?logo=python&logoColor=white&label=python&labelColor=gray&color=blue)
[![Static Badge](https://img.shields.io/badge/DOI-doi.org%2F10.1002%2Fchem.202301043-purple)](https://doi.org/10.1002/chem.202301043)

## 框架 Framework

![Framework](https://github.com/KangruiYuan/SimuBox/blob/main/Docs/Figures/summary.png)


## 安装引导 Installation 

1. From Github

`git clone https://github.com/KangruiYuan/SimuBox.git`

2. From PyPI

相应版本一般落后于`Github`仓库

`pip install SimuBox`

> TIPS: 安装后配置环境推荐使用poetry，依次执行以下python指令
> 
> `pip install poetry`
> 
> `poetry install`

## 使用示例 Demos

示例请见[`/Demos/Scripts.ipynb`](https://github.com/KangruiYuan/SimuBox/blob/main/Demos/Scripts.ipynb)

对于部分功能，进行了网页端可视化，在成功安装之后，可以通过以下指令启动网页服务：

`python -m SimuBox.run`

对于更为全面的功能使用，仍需通过代码实现。

## 联系方式 Contact

> 生活的全部意义在于无穷地探索尚未知道的东西，在于不断地增加更多的知识。 —— 左拉 《萌芽》

该仓库功能相对驳杂，由于迭代问题，编写风格仍有待提高。
任何问题请发布在`issue`或者通过邮件`E-mail：kryuan@qq.com`联系本人。


## 引用 Citing

感谢复旦大学李卫华教授课题组。

```bibtex
@article{SimuBox,
   author = {Yuan, K. and Xu, Z. and Li, W. and Huang, X.},
   title = {Reexamine the emergence and stability of the square cylinder phase in block copolymers},
   journal = {Chem. Eur. J.},
   volume = {10.1002/chem.202301043},
   pages = {e202301043},
   keywords = {block copolymer, self-assembly, self-consistent field theory, square cylinder
phase},
   ISSN = {1521-3765 (Electronic)
0947-6539 (Linking)},
   DOI = {10.1002/chem.202301043},
   url = {https://www.ncbi.nlm.nih.gov/pubmed/37199182},
   year = {2023},
   type = {Journal Article}
}
```