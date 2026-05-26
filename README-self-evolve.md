# Self-evolvable cap-x

设计文档已按功能拆分，迁移到 **[`docs-se/`](docs-se/README.md)**：互不影响的功能各占一个
文件，便于分别实现；共享契约（术语、存储 schema、注入点、配置）单独抽出。

入口与文档地图见 [`docs-se/README.md`](docs-se/README.md)。


01-interactive-loop.md 当我们 click stop trail 然后找一个新的 config start trail, 我们的viser啥都不显示。感觉有逻辑漏洞

我们interactive gui 我希望统一真机实验和模拟器实验。现在模拟器实验会出现reset这件事，我们的真机实验reset是人来做的。给我一个能兼顾两者的交互设计逻辑。我希望真机的reset 会自动调用回rest pose，然后给一个rule-based 的固定流程：1. 通过访问机械臂的接口，循环查看看机械臂是否connect了，循环内 if not: 则第一个弹出的信息就是让user重新启动机械臂服务和中间件等，重启后点这个窗口里的ready按键，然后会再次测试机械臂connect的情况，如果没有的话则下次循环继续问。 如果connect了，则跳出检查机械臂连接的循环，到下一步 2. 循环提示user检查机械臂是否在rest pose，没有的话可以点叉，这个会执行机械臂自动回归rest pose的指令， 然后下一个循环继续问。直到user点对勾，跳出循环，到下一步。3.提示user重新摆放环境，摆放完成后点击对勾。 直到第三步完成，我们的real robot reset才算结束。
