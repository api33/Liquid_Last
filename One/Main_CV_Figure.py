import cv2
import matplotlib.pyplot as plt
import numpy as np
import keyboard


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


def plot_count(Lines):
    lst = Lines
    count_lst = [0] * 256
    for v in lst:
        count_lst[v] += 1
    plt.plot(count_lst)
    plt.ylabel("Count")
    plt.xlabel("Gray")
    plt.xlim(150, 250)
    plt.show()


def funcmain():
    # 读取图像
    #image_path = 'picture/2023_10_24 21-23-19-439.bmp'  # 替换成你的图像路径
    #image_path = 'picture/2023_10_24 21-23-20-296.bmp'  # 替换成你的图像路径
    # image_path = 'picture/20221217165719077.bmp'  # 替换成你的图像路径
    #image_path = 'picture/20221217165722270.bmp'  # 替换成你的图像路径

    image_path='picture1/2023_05_22 17-58-38-391.bmp'

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_copy = image.copy()
    # 获取图像宽度和高度
    height, width = image.shape

    # 选择一个确定的高度
    chosen_height = 860  # 替换成你实际需要的高度值

    gray_2000 = [int(image[h, 2000]) for h in range(height)]
    # print(gray_2000)

    differences = []
    # 计算相邻数据点的差异
    for i in range(len(gray_2000)):
        start = max(0, i - 10)
        differences.append(gray_2000[i] - gray_2000[start])
    idx = differences.index(min(differences))
    gray_values = [image[chosen_height, x] for x in range(width)]

    # 计算每个像素点距离最左侧的距离
    distances = list(range(width))

    # 绘制不同高度的灰度变化曲线
    plt.plot(gray_2000)
    plt.ylabel('Gray')
    plt.xlabel('Height')
    # plt.show()

    # 绘制差异图
    plt.plot(differences)
    plt.xlabel('height')
    plt.ylabel('Difference')
    # plt.show()

    color = (0, 0, 255)  # 线的颜色，这里使用RGB格式 (0, 255, 0) 表示绿色
    thickness = 2  # 线的宽度
    dist = 120
    left_boundle = 1750
    right_boundle = 2250
    cv2.line(image, (left_boundle, idx - dist), (right_boundle, idx - dist), color, thickness)
    cv2.line(image, (left_boundle, idx + dist), (right_boundle, idx + dist), color, thickness)
    cv2.line(image, (left_boundle, idx - dist), (left_boundle, idx + dist), color, thickness)
    cv2.line(image, (right_boundle, idx - dist), (right_boundle, idx + dist), color, thickness)
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

    print(count1, count2, len(Rectongle_Points))
    plt.plot(average_lines)
    plt.ylabel('Average')
    plt.xlabel('Height')
    print(len(Points_lines))
    # 绘制矩形的两个角点
    plt.scatter(x_points, y_points, color='red', label='Points')
    plt.scatter(other_points_x1, other_points_y1, color='green', label='Points')
    plt.scatter(other_points_x2, other_points_y2, color='blue', label='Points')
    plt.plot([0, 225], [midpoint_y, midpoint_y], marker='o')
    plt.plot([midpoint_x, midpoint_x], [50, 200], marker='o')
    # plt.show()


    # while True:
    #     key = cv2.waitKey(0)
    #
    #     # 如果按下的是 Enter 键 (ASCII码为 13)
    #     if key == 13:
    #         # 画一条线，这里示例是在图像中间画一条横线
    #         cv2.line(image, (0, image.shape[0] // 2), (image.shape[1], image.shape[0] // 2), (0, 255, 0), 2)
    #
    #         # 显示带有新线的图像
    #         cv2.imshow('Adjusted Window', image)
    #
    #     # 如果按下的是 Esc 键 (ASCII码为 27)，则退出循环
    #     elif key == 27:
    #         break

    window_width, window_height = 1000, 800
    # #展示图像窗口
    cv2.namedWindow('Adjusted Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Adjusted Window', window_width, window_height)

    i_lines = 0
    i=0
    while True:
        user_input = input("按下空格键结束循环：")

        if user_input == "":
            lst = Points_lines[i_lines]
            count_lst = [0] * 256
            for v in lst:
                count_lst[v] += 1
            #plt.xlim(100, 250)
            plt.plot(count_lst)
            i_lines += 1
            plt.draw()
            plt.pause(0.0001)
            plt.clf()

            if i_lines > len(Points_lines) - 1:
                print("超出", i_lines)
                break
            image1 = image.copy()

            # 在图像上绘制一条线
            cv2.line(image1, (0, idx-dist+i_lines), (image1.shape[1], idx-dist+i_lines), (0, 255, 0), 2)
            # 显示带有新线的图像
            i += 1
            cv2.imshow('Adjusted Window', image1)
            cv2.waitKey(3)  # 使用非零参数，等待1毫秒，使得图像能够及时更新

        elif user_input == " ":
            break

    window_width, window_height = 1000, 800
    # #展示图像窗口
    cv2.namedWindow('Adjusted Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Adjusted Window', window_width, window_height)
    cv2.imshow('Adjusted Window', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


funcmain()