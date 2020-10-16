import pprint
import cv2
import numpy as np
from PIL import Image
import math
from random import randint
import os

# 上黄右橘前绿下白左红后蓝
TILE_NUM = 8
PIX_NUM = 9
PIX_VAR = 4
WIDTH = 256


def select_pix_array():
    """
    6面，2是x和y，16是2*8
    一个面的，有8个方块需要识别，每个需要n个像素点，每个像素点x和y
    16*9*2 16方块，9个样本
    tile*pix*cordi
    """

    x = [17, 50, 83,
         17,     83,
         17, 50, 83]
    y = [17, 17, 17,
         50,     50,
         83, 83, 83]
    a, b, c = 2, PIX_NUM, TILE_NUM
    local_pix = [
        [
            [0 for col in range(a)]
            for col in range(b)
        ]
        for row in range(c)
    ]

    for tile in range(TILE_NUM):
        # 第tile块的PIX_NUM个样本
        for sample in range(PIX_NUM):
            local_pix[tile][sample][0] = int(WIDTH *
                                             x[tile] / 100) + randint(-PIX_VAR, PIX_VAR)
            local_pix[tile][sample][1] = int(WIDTH *
                                             y[tile] / 100) + randint(-PIX_VAR, PIX_VAR)
    print('local_pix')
    pprint.pprint(local_pix)

    return local_pix


def average_hsv_of_each_tile(local_pix, rgb):
    w, h = 3, 16  # 块数
    hsv_array_per_camera = [[0 for x in range(w)] for y in range(TILE_NUM)]

    # 16 square
    for tile in range(0, TILE_NUM):
        # 9 samples
        Sample0 = rgb[local_pix[tile][0][1], local_pix[tile][0][0]]
        Sample1 = rgb[local_pix[tile][1][1], local_pix[tile][1][0]]
        Sample2 = rgb[local_pix[tile][2][1], local_pix[tile][2][0]]
        Sample3 = rgb[local_pix[tile][3][1], local_pix[tile][3][0]]
        Sample4 = rgb[local_pix[tile][4][1], local_pix[tile][4][0]]
        Sample5 = rgb[local_pix[tile][5][1], local_pix[tile][5][0]]
        Sample6 = rgb[local_pix[tile][6][1], local_pix[tile][6][0]]
        Sample7 = rgb[local_pix[tile][7][1], local_pix[tile][7][0]]
        Sample8 = rgb[local_pix[tile][8][1], local_pix[tile][8][0]]

        for j in range(0, 3):
            # average
            hsv_array_per_camera[tile][j] = (int(Sample0[j]) + int(Sample1[j]) + int(Sample2[j]) + int(Sample3[j]) + int(
                Sample4[j]) + int(Sample5[j]) + int(Sample6[j]) + int(Sample7[j]) + int(Sample8[j])) / 9

        temp = cv2.cvtColor(
            np.uint8([[hsv_array_per_camera[tile]]]), cv2.COLOR_BGR2HSV)
        hsv_array_per_camera[tile] = temp[0][0]

    return hsv_array_per_camera


def color_rec(faces, hsv_values):
    color = str()

    # 设置颜色范围
    #-------------------|-------U---------|---------R---------|---------F---------|--------D--------|--------L--------|---------B---------|
    V_threshold_all = [135,  # U
                       150,  # R
                       138,  # F
                       140,  # D
                       140,  # L
                       130]  # B
    cold_H_range_all = [[35, 95, 145],
                        [35, 95, 145],
                        [35, 95, 145],
                        [35, 95, 145],
                        [35, 95, 145],
                        [35, 95, 145]]  # red, green, blue
    warm_H_range_all = [[25, 52, 95, 145],
                        [25, 52, 95, 145],
                        [25, 52, 95, 145],
                        [25, 52, 95, 145],
                        [25, 52, 95, 145],
                        [25, 52, 95, 145]]  # orange, yellow, green, blue
    S_min_all = [40,
                 40,
                 40,
                 20,
                 15,
                 40]

    cold_H_range = []
    warm_H_range = []

    # S_min = 40
    if faces == "U":
        cold_H_range = cold_H_range_all[0]
        warm_H_range = warm_H_range_all[0]
        S_min = S_min_all[0]
        V_threshold = V_threshold_all[0]
    elif faces == "R":
        cold_H_range = cold_H_range_all[1]
        warm_H_range = warm_H_range_all[1]
        S_min = S_min_all[1]
        V_threshold = V_threshold_all[1]
    elif faces == "F":
        cold_H_range = cold_H_range_all[2]
        warm_H_range = warm_H_range_all[2]
        S_min = S_min_all[2]
        V_threshold = V_threshold_all[2]
    elif faces == "D":
        cold_H_range = cold_H_range_all[3]
        warm_H_range = warm_H_range_all[3]
        S_min = S_min_all[3]
        V_threshold = V_threshold_all[3]
    elif faces == "L":
        cold_H_range = cold_H_range_all[4]
        warm_H_range = warm_H_range_all[4]
        S_min = S_min_all[4]
        V_threshold = V_threshold_all[4]
    else:  # B
        cold_H_range = cold_H_range_all[5]
        warm_H_range = warm_H_range_all[5]
        S_min = S_min_all[5]
        V_threshold = V_threshold_all[5]

    S_orange = 185
    H_orange = 110

    for n in range(0, 9):
        if hsv_values[n][1] >= S_min:
            if hsv_values[n][2] >= V_threshold:
                # warm
                # ORANGE
                if hsv_values[n][0] <= warm_H_range[0]:
                    value = "O"
                # YELLOW
                elif hsv_values[n][0] <= warm_H_range[1]:
                    value = "Y"
                # GREEN lower priority )
                elif hsv_values[n][0] <= warm_H_range[2]:
                    value = "G"
                # BLUE(lower priority )
                elif hsv_values[n][0] <= warm_H_range[3]:
                    value = "B"
                # RED(lower priority )
                else:
                    value = "R"

            else:
                # cold
                # RED
                if (hsv_values[n][0] <= cold_H_range[0] or hsv_values[n][0] >= cold_H_range[2]):
                    value = "R"
                    if (hsv_values[n][1] >= S_orange) and (hsv_values[n][2] >= H_orange):
                        value = "O"
                # GREEN
                elif (hsv_values[n][0] <= cold_H_range[1]):
                    value = "G"
                # BLUE
                else:
                    value = "B"
        else:
            # WHITE
            value = "W"

        print('{} {} {} {}'.format(value, str(hsv_values[n][0]), str(
            hsv_values[n][1]), str(hsv_values[n][2])), end="   ")
        color = color + value

    print("\n")
    return color


def formatstr(in_string):
    # 上黄右橘前绿下白左红后蓝
    output = ""
    seq_pos = ["U", "R", "F", "D", "L", "B"]  # position
    seq_col = ["Y", "O", "G", "W", "R", "B"]  # color
    for item in in_string:
        for i in range(0, len(seq_pos)):
            if item == seq_col[i]:
                output = output + str(seq_pos[i])
    # print(output)
    return (output)


def scancubemain():
    FILEPATH = os.getcwd()
    rgb_F = cv2.imread(  # G
        FILEPATH + "/output/Right_green.jpg")
    rgb_L = cv2.imread(  # red
        FILEPATH + "/output/Left_red.jpg")
    rgb_U = cv2.imread(  # yellow
        FILEPATH + "/output/Top_yellow.jpg")
    rgb_D = cv2.imread(  # white
        FILEPATH + "/output/Top_white.jpg")
    rgb_R = cv2.imread(  # orange
        FILEPATH + "/output/Right_orange.jpg")
    rgb_B = cv2.imread(  # blue
        FILEPATH + "/output/Left_blue.jpg")
    # 256*256

    pix = select_pix_array()
    # Front
    f_hsv_array = average_hsv_of_each_tile(pix, rgb_F)
    f_green_hsv_values = f_hsv_array
    f_green_hsv_values.insert(4, [80, 255, 10])
    # Left
    l_hsv_array = average_hsv_of_each_tile(pix, rgb_L)
    l_red_hsv_values = l_hsv_array
    l_red_hsv_values.insert(4, [0, 255, 10])
    # Up
    u_hsv_array = average_hsv_of_each_tile(pix, rgb_U)
    u_yellow_hsv_values = u_hsv_array
    # insert default color at index 4
    u_yellow_hsv_values.insert(4, [50, 255, 255])

    # Down
    d_hsv_array = average_hsv_of_each_tile(pix, rgb_D)
    d_white_hsv_values = d_hsv_array
    d_white_hsv_values.insert(4, [0, 0, 255])

    # Right
    r_hsv_array = average_hsv_of_each_tile(pix, rgb_R)
    r_orange_hsv_values = r_hsv_array
    r_orange_hsv_values.insert(4, [5, 255, 255])

    # Back
    b_hsv_array = average_hsv_of_each_tile(pix, rgb_B)
    b_blue_hsv_values = b_hsv_array
    b_blue_hsv_values.insert(4, [120, 255, 10])

    print("Start recognising colors...\n")

    output_string = color_rec(
        "U", u_yellow_hsv_values) + "\n"  # Y  #up pic upper
    output_string = output_string + \
        color_rec("R", r_orange_hsv_values) + "\n"  # O  #up pic bottom
    output_string = output_string + \
        color_rec("F", f_green_hsv_values) + "\n"  # G  #side pic bottom
    output_string = output_string + \
        color_rec("D", d_white_hsv_values) + "\n"  # W  #down pic bottom
    output_string = output_string + \
        color_rec("L", l_red_hsv_values) + "\n"  # R  #side pic upper
    output_string = output_string + \
        color_rec("B", b_blue_hsv_values) + "\n"  # B  #down pic upper

    print(output_string)
    print(formatstr(output_string))

    return formatstr(output_string)

if __name__ == '__main__':
    scancubemain()
