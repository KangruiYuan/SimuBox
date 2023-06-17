
# *SimuBox* for Science Calculation

GitHub Version: ***0.1.7***

PYPI Version: ***0.1.6***

该库为本人求学期间为完成科研工作而常用的模块的集合，可能并不完善。若有好意见请发布在`issue`或者通过邮件联系本人。
`e-mail：kryuan@qq.com`

# `Guide for Installment` 安装引导

## 请使用以下指令进行安装或者从`GitHub`中下载

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

# `Usage` 使用说明

## `Modules & Functions` 模块功能

- *SciTools*
  - **reader**
    - InfoReader: 读取各类输出信息
  - **generator**
    - TopoCreater: 生成分子拓扑及实现RPA计算
    - PhiCreater (Coming soon...)
  - **xmltrans**
    - XmlTransformer: 将 `galamost` 生成的 `.xml` 文件转为密度文件
- *SciCalc*
  - **scatter**
    - Scatter: 模拟散射
  - **voronoi**
    - VoronoiCell: 基于密度对Voronoi Cell进行划分
  - **correlation**
    - Corr: 计算关联函数
- *SciPlot*
  - **compare**
    - CompareJudger: 比较绘图
    - Labels: 标签集合
  - **land**
    - Landscaper: 绘制二维登高线图
  - **phase**
    - PhaseDiagram: 绘制相图
- ~~**ScriptRun**~~
  
  该模块暂不进行公开。

[//]: # (  以下脚本仅在`GitHub`上传，并未包含在PyPI的版本中。根据不同的需求，需要对以下脚本进行特定的补充和修改。)

[//]: # ()
[//]: # (  - push_job_TOPS)

[//]: # (  - repush)

[//]: # (  - template)

[//]: # (  - lib_tools)

[//]: # (  - extract_data)

