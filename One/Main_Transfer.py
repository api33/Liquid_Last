import cv2
import matplotlib.pyplot as plt
import numpy as np

import os
from scipy.signal import find_peaks

from One.Part_Move import func_move
from One.Part_Transfer import funcTransfer
from One.test3 import func_find_four


def generate_approximate_curve(points):
    x, y = zip(*points)
    x = np.array(x)
    y = np.array(y)
    # 生成二次多项式拟合曲线
    coefficients = np.polyfit(x, y, deg=7)
    poly = np.poly1d(coefficients)

    midpoint_x = np.mean(x)
    # 计算几何中点的 y 坐标
    midpoint_y = poly(midpoint_x)
    return (int(midpoint_x), int(midpoint_y))



def find_top_four_peaks(lst,Allpoints):
    peaks, _ = find_peaks(lst)

    # 获取波峰的位置和高度
    peak_positions = peaks
    peak_heights = [lst[peak] for peak in peaks]

    # 根据波峰高度降序排序
    sorted_peaks = sorted(zip(peak_positions, peak_heights), key=lambda x: x[1], reverse=True)

    # 获取最高的四个波峰
    top_four_peaks = sorted_peaks[:4]
    a = top_four_peaks[0][0]
    b = top_four_peaks[2][0]
    if abs(b - a) < 10:
        top_four_peaks = sorted_peaks[:2] + sorted_peaks[4:6]
    top_pos=[pos for pos,_ in top_four_peaks]
    #top_pos.sort()
    new_top = sorted(top_pos)

    for i in range(4):
        temp = new_top[i]

        if i == 1:
            currmin = min(Allpoints[temp:temp + 30])
            while Allpoints[temp] > currmin+100:
                temp += 1
            new_top[1] = temp
            continue
        if i == 3:
            currmin = min(Allpoints[temp:temp + 30])
            while Allpoints[temp] > currmin+100:
                temp += 1
            new_top[3] = temp

    return new_top
def funcmain(image_add,filename):
    # 读取图像
    image_path=image_add


    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_copy = image.copy()
    # 获取图像宽度和高度
    height, width = image.shape

    # 选择一个确定的高度89
    chosen_height = 860  # 替换成你实际需要的高度值

    gray_2000 = [int(image[h, width//2]) for h in range(height)]


    differences = []
    # 计算相邻数据点的差异
    for i in range(len(gray_2000)):
        start = max(0, i - 10)
        differences.append(gray_2000[i] - gray_2000[start])
    idx = differences.index(min(differences[:height-300]))

    gray_values = [image[chosen_height, x] for x in range(width)]
    # 计算每个像素点距离最左侧的距离
    distances = list(range(width))


    dist = 120
    left_boundle = width//2-250
    right_boundle = width//2+250
    Rectongle_Points = []
    Points_lines = [[] for _ in range(2 * dist + 1)]
    x_points = []
    y_points = []
    count = 0
    for h in range(idx - dist, idx + dist + 1):
        for w in range(left_boundle, right_boundle + 1):
            if count < 20:
                count += 1
            x_points.append(h - (idx - dist))
            y_points.append(image_copy[h, w])
            Rectongle_Points.append((h - (idx - dist), image_copy[h, w]))
            Points_lines[h - (idx - dist)].append(image_copy[h, w])

    average_lines = [sum(v) // len(v) for v in Points_lines]
    points = [(i, v) for i, v in enumerate(average_lines)]
    midpoint_x, midpoint_y = generate_approximate_curve(points)
    count1 = 0
    count2 = 0
    other_points_x1 = []
    other_points_y1 = []
    other_points_x2 = []
    other_points_y2 = []
    for x, y in Rectongle_Points:
        if x < midpoint_x and y < midpoint_y:
            count1 += 1
            other_points_x1.append(x)
            other_points_y1.append(y)
        if x > midpoint_x and y > midpoint_y:
            other_points_x2.append(x)
            other_points_y2.append(y)
            count2 += 1


    Points_lines_copy=[[] for _ in range(2 * dist + 1)]
    for i in range(2 * dist + 1):
        for v in Points_lines[i]:
            Points_lines_copy[i].append(int(v))
    #print(Points_lines_copy[0])
    Points_lines_compare=[[] for _ in range(2 * dist + 1)]
    for i in range(2*dist+1):
        Points_lines_compare[i]=Points_lines_copy[i][:]

    Varian_lst=[]
    Std_lst=[]
    Mean_lst=[]
    for i in range(2 * dist + 1):
        mean_value = np.mean(Points_lines_copy[i])
        std_deviation_value = np.std(Points_lines_copy[i])
        # 计算方差
        Mean_lst.append(mean_value)
        variance_value = np.var(Points_lines_copy[i])
        Varian_lst.append(variance_value)
        Std_lst.append(std_deviation_value)

    # plt.plot(Std_lst)
    # plt.ylabel('Old_Std')
    # plt.xlabel('Height')
    # plt.xlim(0, 241)
    # plt.xticks(range(0, 241, 20))
    # plt.show()

    AllPoints=[0]*256
    for h in range(idx - dist, idx + dist + 1):
        for w in range(left_boundle, right_boundle + 1):
            point_gray = image_copy[h, w]
            AllPoints[point_gray] += 1

    Countdifference=[]
    for i in range(len(AllPoints)-10):
        diff= AllPoints[i+10]-AllPoints[i]
        Countdifference.append(abs(diff))

    lst1=func_find_four(AllPoints)
    # print(lst1,AllPoints)

    # plt.plot(AllPoints)
    # plt.ylabel('Before')
    # plt.xlabel(filename)
    # plt.show()
    Transfer_nums=func_move(AllPoints,lst1)
    #print(Points_lines_copy[0][:10])
    New_Points_lines=funcTransfer(Transfer_nums,Points_lines_copy,lst1)
    #print(New_Points_lines[0][:10])
    Std_lst = []
    for i in range(2 * dist + 1):
        std_deviation_value = np.std(New_Points_lines[i])
        Std_lst.append(std_deviation_value)
    #print(np.trapz(Std_lst))


    #print(Transfer_nums)
    #print("Old180",Points_lines_compare[180])

    #print("New180",New_Points_lines[180])
    #print(np.std(Points_lines_compare[180][:10]),np.std(New_Points_lines[180][:10]))

    plt.plot(Std_lst)
    plt.ylabel('New_Std')
    plt.xlabel('Height')
    plt.xlim(0, 241)
    plt.xticks(range(0, 241, 20))
    plt.show()

    return np.trapz(Std_lst)













#image_add='picture2/2023_09_12 04-45-43-365.bmp'
#image_add='picture2/2023_09_12 04-44-00-279.bmp'
# image_add='picture2/2023_09_12 04-57-37-023.bmp'


def Main_Move():
    file_names = []
    # 遍历目录下的所有文件
    for filename in os.listdir('picture3'):
        # 检查是否是文件而不是子目录
        if os.path.isfile(os.path.join('picture3', filename)):
            file_names.append(filename)
    count=0

    All_Std_intergrate=[]
    for filename in file_names[1190:1195]:
        print(f"第{count}张图")
        res=funcmain('picture3/'+filename,filename)
        if res>1300:
            print(res,filename)
        if res<900:
            print(res,filename)
        All_Std_intergrate.append(res)
        count += 1
    print(All_Std_intergrate)

    plt.scatter([i for i in range(len(All_Std_intergrate))],All_Std_intergrate)
    plt.ylabel('All_Std_intergrate')
    plt.xlabel('Picture')
    plt.show()
Main_Move()




