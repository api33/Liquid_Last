import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
import keyboard
import heapq

from scipy.signal import find_peaks
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
    if abs(b - a) <=10:
        print("fddfd")
        top_four_peaks = sorted_peaks[:2] + sorted_peaks[4:6]
    top_pos=[pos for pos,_ in top_four_peaks]
    #top_pos.sort()
    new_top = sorted(top_pos)
    print(new_top)
    for i in range(4):
        temp = new_top[i]

        if i == 1:
            print(temp)
            currmin = min(Allpoints[temp:temp + 30])+300
            while Allpoints[temp] > currmin:
                temp += 1
            new_top[1] = temp
            continue
        if i == 3:
            print(temp)
            currmin = min(Allpoints[temp:temp + 30])
            while Allpoints[temp] > currmin:
                temp += 1
            new_top[3] = temp
    print(new_top,Allpoints[57],Allpoints[92],Allpoints[66])
    #return [pos for pos,_ in top_four_peaks]
    return new_top
    #return top_pos
def funcmain(image_add):
    # 读取图像
    #image_path = 'picture/2023_10_24 21-23-19-439.bmp'  # 替换成你的图像路径
    #image_path = 'picture/2023_10_24 21-23-20-296.bmp'  # 替换成你的图像路径
    #image_path = 'picture/20221217165719077.bmp'  # 替换成你的图像路径
    #image_path = 'picture/20221217165722270.bmp'  # 替换成你的图像路径



    #picture1
    image_path='picture1/2023_05_22 17-58-38-391.bmp'
    #image_path='picture1/2023_05_22 17-59-31-290.bmp'
    image_path='picture1/2023_05_22 18-01-01-271.bmp'
    image_path='picture1/2023_09_12 04-45-42-351.bmp'
    #image_path='picture1/2023_09_12 04-49-50-142.bmp'

    #picture2
    image_path='picture2/2023_09_12 04-45-43-365.bmp'
    #image_path='picture2/2023_09_12 04-47-25-095.bmp'
    #image_path='picture2/2023_09_12 04-45-44-865.bmp'
    image_path=image_add


    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_copy = image.copy()
    # 获取图像宽度和高度
    height, width = image.shape

    # 选择一个确定的高度
    chosen_height = 860  # 替换成你实际需要的高度值

    gray_2000 = [int(image[h, width//2]) for h in range(height)]
    # print(gray_2000)

    differences = []
    # 计算相邻数据点的差异
    for i in range(len(gray_2000)):
        start = max(0, i - 10)
        differences.append(gray_2000[i] - gray_2000[start])
    idx = differences.index(min(differences[:height-300]))
    # idx=870
    print("idx",idx)
    gray_values = [image[chosen_height, x] for x in range(width)]

    # 计算每个像素点距离最左侧的距离
    distances = list(range(width))

    # 绘制不同高度的灰度变化曲线
    # plt.plot(gray_2000)
    # plt.ylabel('Gray')
    # plt.xlabel('Height')
    # plt.show()

    # 绘制差异图
    # plt.plot(differences)
    # plt.xlabel('height')
    # plt.ylabel('Difference')
    # plt.show()

    color = (0, 255, 0)  # 线的颜色，这里使用RGB格式 (0, 255, 0) 表示绿色
    thickness = 2  # 线的宽度
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
                # print(h)
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
    print(Std_lst)
    print(len(Std_lst))
    flag=False
    print("标准差曲线面积",np.trapz(Std_lst))

    AllPoints=[0]*256
    # for h in range(height):
    #     for w in range(width):
    #         point_gray=image_copy[h,w]
    #         AllPoints[point_gray]+=1
    for h in range(idx - dist, idx + dist + 1):
        for w in range(left_boundle, right_boundle + 1):
            point_gray = image_copy[h, w]
            AllPoints[point_gray] += 1
    print(AllPoints)
    Countdifference=[]
    for i in range(len(AllPoints)-10):
        diff= AllPoints[i+10]-AllPoints[i]
        Countdifference.append(abs(diff))
    print(Countdifference)
    lst1=find_top_four_peaks(Countdifference,AllPoints)
    plt.plot(Countdifference)
    plt.ylabel('Count')
    plt.xlabel('Gray')
    # plt.xlim(0, 241)
    # plt.xticks(range(0, 241, 20))
    plt.show()
    plt.plot(AllPoints)
    # lst1=[62,76,137,172]
    lst2=[]
    for v in lst1:
        lst2.append(AllPoints[v])
    plt.scatter(lst1,lst2,c='red', marker='o', label='Data Points')
    plt.ylabel('Count')
    plt.xlabel('Gray')
    # plt.xlim(0, 241)
    # plt.xticks(range(0, 241, 20))
    plt.show()

    a,b,c,d=lst1
    part1,part2=[],[]
    for i in range(a,b+1):
        part1.append((i,AllPoints[i]))
    for i in range(c,d+1):
        part2.append((b+i-c,AllPoints[i]))

    plt.plot([x for x,_ in part1+part2],[y for _,y in part1+part2])
    #plt.plot([x for x,_ in part2],[y for _,y in part2])
    plt.ylabel('Std')
    plt.xlabel('Height')
    plt.xlim(0,241)
    plt.xticks(range(0, 241, 20))
    plt.show()

    cv2.line(image, (left_boundle, idx - dist), (right_boundle, idx - dist), color, thickness)
    cv2.line(image, (left_boundle, idx + dist), (right_boundle, idx + dist), color, thickness)
    cv2.line(image, (left_boundle, idx - dist), (left_boundle, idx + dist), color, thickness)
    cv2.line(image, (right_boundle, idx - dist), (right_boundle, idx + dist), color, thickness)


    # print(idx,idx-dist)
    original_height, original_width = image.shape[:2]
    window_width, window_height = 600, 500
    # 定义窗口的目标大小（比例可以根据需求调整）
    target_width = 600
    target_height = int((target_width / original_width) * original_height)

    # #展示图像窗口
    # cv2.namedWindow('Adjusted Window', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Adjusted Window', target_width, target_height)
    # cv2.imshow('Adjusted Window', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

image_add1='picture3/2023_09_12 04-46-47-954.bmp'
image_add2='picture3/2023_09_12 04-51-47-273.bmp'
funcmain(image_add1)
funcmain(image_add2)


[53, 69, 110, 157]

[53,69]
[49,64]
[49, 64, 72, 107]
