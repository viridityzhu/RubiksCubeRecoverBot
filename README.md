# RubiksCubeRecoverBot 魔方复原机器人
A robot to recover Rubik's cube, based on Arduino, using Python combined with Machine Learning and OpenCV. Thanks to these projects and resources:

一个基于Arduino的魔方复原机器人，使用Python编写，结合Machine Learning和OpenCV。 感谢这些项目：

- https://github.com/krustnic/RubikKeras
- http://kociemba.org/cube.htm
- https://github.com/g20150120/cubot
- https://github.com/hkociemba/RubiksCube-TwophaseSolver

## One. Software Part
The function of this robot is divided into two parts: **Rubik's cube recognition** and **automatically recover it**.

The first part focuses on recognizing and solving Rubik's cube, which is mainly written in Python. The second part focuses on controlling the Arduino to move the Rubik's cube, which is written in a `.ino` Arduino program.

为实现魔方识别与自动复原，我们将程序主要分为两部分。

第一部分实现魔方的识别及求解功能，主要用Python编写；第二部分实现控制Arduino驱动步进电机实现转动魔方，主要实现为.ino格式的单个Arduino程序。

### Part 1: Recognizing and solve the Rubik's cube 魔方识别与求解

Written in Python. There are 5 steps:

1. Using Keras to build up a Unet model, and carve up the 6 faces of the Rubik's cube.
2. Using OpenCV to parse the colors of the bricks.
3. A GUI to show the colors, allowing correcting the color.
4. Solving the Rubik's cube via Two-phase algorithm.
5. Sending the solution to the Arduino via serial port.

本部分程序采用Python实现。我们将魔方识别与求解的步骤更具体地分为五个部分：

1. 采用Keras库建立Unet模型，实现从摄像头图像中分割魔方的6个面。感谢[RubikKeras项目](https://github.com/krustnic/RubikKeras)提供的模型。
2. 采用OpenCV，解析魔方块的颜色
3. 以GUI界面显示魔方色块，提供手工校正魔方色块的功能
4. 用Two-phase算法求出魔方最优解
5. 将魔方解以字符串形式通过串口传给Arduino板


