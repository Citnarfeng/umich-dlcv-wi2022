# Deep Learning for Computer Vision · WI2022

密歇根大学课程学习与作业仓库。日常流程：**GitHub 打开 → Colab 编写并运行 → 保存回 GitHub 同一路径**。

[课程安排](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/schedule.html)

## 开始 Assignment 1

| 顺序 | 作业 | Colab 入口 | 保存到 GitHub 的文件路径 |
| --- | --- | --- | --- |
| 1 | PyTorch 101 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Citnarfeng/umich-dlcv-wi2022/blob/main/colab/a1/pytorch101.ipynb) | `colab/a1/pytorch101.ipynb` |
| 2 | k-Nearest Neighbors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Citnarfeng/umich-dlcv-wi2022/blob/main/colab/a1/knn.ipynb) | `colab/a1/knn.ipynb` |

两个 Notebook 均包含必要的实现代码和辅助模块，准备阶段不需要 Drive、Git clone 或访问 GitHub 的令牌。CIFAR-10 仍会在运行相应作业单元格时下载；运行时文件和数据集不由“保存 Notebook”持久化。

## 首次连接私有仓库

1. 登录你的 Google 账号，打开 [Colab 的 GitHub 浏览器](https://colab.research.google.com/github)。
2. 勾选 **Include private repos**，在 GitHub 授权窗口登录 `Citnarfeng` 并授权 Colab。首次保存也可能要求写入授权。
3. 搜索 `Citnarfeng/umich-dlcv-wi2022`，选择 `main`，打开 `colab/a1/` 下的 Notebook；授权后也可使用上面的按钮。

如果入口提示找不到 Notebook，先检查是否登录正确 GitHub 账号、已允许 Colab 访问私有仓库。无需把仓库改为 Public，也无需在代码里填写密钥。

## 每次学习

1. 从本页入口打开 GitHub 上最新的 Notebook。需要 GPU 时，在 Colab 的运行时设置中选择 GPU。
2. 按顺序运行准备部分。到 **编辑作业实现** 的 `%%writefile … .py` 单元格中填写 TODO。
3. 修改实现后，重新运行该写文件单元格，然后运行相应题目的测试。**不要只在左侧文件栏修改 .py**；这些改动不会自动进入 Notebook。
4. 选择 **File → Save a copy in GitHub**（部分界面为 Save a copy to GitHub）。仓库选 `Citnarfeng/umich-dlcv-wi2022`，分支选 `main`，路径严格使用上表的同名路径。填写简短提交说明，如 `a1: implement tensor slicing`。
5. 到 GitHub 确认该文件出现新提交。下次从 GitHub 重新打开即可接着学习。普通保存快捷键不能作为已提交 GitHub 的确认。

“Save a copy”使用同一仓库、分支和路径时会更新该文件，不必每次另起文件名。两个 Notebook 要分别保存。避免在多个 Colab 标签页同时编辑同一文件并交替保存。

## .py 与 .ipynb 的关系

原作业不是重复的两种格式：`.ipynb` 包含说明、实验和测试；`.py` 定义需要实现的函数。原 Notebook 导入这些函数，所以不能删除实现文件而不作适配。

`colab/a1/` 是日常编辑入口：原 .py 内容已嵌入 `%%writefile` 单元格。运行时生成同名文件供原测试导入，保存 Notebook 就会保留实现。`assignments/a1/` 是本次接入时的原始快照，不会自动随 Colab 修改而更新；需要独立 .py 时使用 Notebook 底部的导出函数。

原始作业要求提交时保持指定结构。Colab 适配版调整了启动部分，不直接作为已经符合课程要求的提交包；正式提交前应将答案整理回原模板并运行所需测试。

## 文件与来源

```text
colab/a1/         # 两个日常使用的 Colab Notebook
assignments/a1/   # 本地 Assignment1 原始快照（Notebook、实现模块、eecs598）
```

当前文件来自本机已有的 `~other/Assignment1/`，接入过程未修改该目录。原文标注 EECS 498-007/598-005，并含旧学期路径示例；尚未核实这份素材与 WI2022 发布包完全一致。仓库按用户的 WI2022 学习目标命名，保留原题标题与来源差异。

进度：已接入工作流，尚未完成作业 TODO；未执行整份作业或训练。仅验证文件结构、嵌入源码一致性及准备阶段写文件行为。首次真实 Colab 会话的登录授权和 GPU/依赖兼容性需在 Colab 中确认。

[Google 官方：Colab 与 GitHub 的打开、私有仓库授权和保存说明](https://github.com/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)
