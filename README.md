# RubiksCubeRecoverBot
A robot to recover Rubik's cube, based on Arduino, using Python combined with Machine Learning and OpenCV. Thanks to these projects and resources:

一个基于Arduino的魔方复原机器人，使用Python编写，结合Machine Learning和OpenCV。 感谢这些项目：

- https://github.com/krustnic/RubikKeras
- http://kociemba.org/cube.htm
- https://github.com/g20150120/cubot
- https://github.com/hkociemba/RubiksCube-TwophaseSolver

## One. Software Part
The function of this robot is divided into two parts: **Rubik's cube recognition** and **automatically recover it**.

为实现魔方识别与自动复原，我们将程序主要分为两部分。

第一部分实现魔方的识别及求解功能，主要用Python编写；第二部分实现控制Arduino驱动步进电机实现转动魔方，主要实现为.ino格式的单个Arduino程序。

3.2.1.第一部分 魔方识别与求解

本部分程序采用Python实现。我们将魔方识别与求解的步骤更具体地分为五个部分：

1. 采用Keras机器学习库，建立并训练深度学习Unet模型，实现从摄像头图像中分割魔方的6个面
2. 采用OpenCV库，解析魔方块的颜色
3. 采用Tkinter库，以可视化图形界面显示魔方色块，提供手工校正魔方色块的功能
4. 用Two phase算法求出魔方最优解
5. 采用Serial库，将魔方解以字符串形式通过串口传给Arduino板


