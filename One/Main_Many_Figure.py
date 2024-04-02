import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
import keyboard
import os


def showfigure(file):

    image_path = 'picture2/' + file
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape

    gray_2000 = [int(image[h, width // 2]) for h in range(height)]

    dist = 120
    left_boundle = width // 2 - 250
    right_boundle = width // 2 + 250
    differences = []
    # 计算相邻数据点的差异
    for i in range(len(gray_2000)):
        start = max(0, i - 10)
        differences.append(gray_2000[i] - gray_2000[start])
    idx = differences.index(min(differences[:height - 300]))




    AllPoints = [0] * 256
    # for h in range(height):
    #     for w in range(width):
    #         point_gray=image_copy[h,w]
    #         AllPoints[point_gray]+=1
    for h in range(idx - dist, idx + dist + 1):
        for w in range(left_boundle, right_boundle + 1):
            point_gray = image[h, w]
            AllPoints[point_gray] += 1
    plt.figure(num=file.split('.')[0])
    plt.plot(AllPoints)

    plt.suptitle(file, fontsize=16)
    plt.ylabel('Count')
    plt.xlabel('Gray')
    plt.savefig('picture2_copy/'+file.split('.')[0]+'.png')
    # plt.xlim(0, 241)
    # plt.xticks(range(0, 241, 20))

    #plt.show()


def get_file_names(directory_path):
    file_names = []

    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        # 检查是否是文件而不是子目录
        if os.path.isfile(os.path.join(directory_path, filename)):
            file_names.append(filename)

    return file_names
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




def funcmain():
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
    # image_path='picture2/2023_09_12 04-45-42-857.bmp'
    # image_path='picture2/2023_09_12 04-45-43-365.bmp'
    #image_path='picture2/2023_09_12 04-45-56-508.bmp'
    # image_path='picture2/2023_09_12 04-47-27-121.bmp'
    image_path='picture2/2023_09_12 04-45-47-390.bmp'


    # 指定目录路径
    directory_path = 'picture2'
    # 调用函数获取文件名列表
    file_names_list = get_file_names(directory_path)
    print(file_names_list)




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

    # 计算每个像素点距离最左侧的距离
    distances = list(range(width))

    # 绘制不同高度的灰度变化曲线
    # plt.plot(gray_2000)
    # plt.ylabel('Gray')
    # plt.xlabel('Height')
    # plt.show()

    # 绘制差异图
    # plt.plot(differences)
    #
    # plt.xlabel('height')
    # plt.ylabel('Difference')
    # plt.show()

    color = (0, 255, 0)  # 线的颜色，这里使用RGB格式 (0, 255, 0) 表示绿色
    thickness = 2  # 线的宽度
    dist = 120
    left_boundle = width//2-250
    right_boundle = width//2+250
    count=0
    for file in file_names_list:
        showfigure(file)
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
    # plt.plot(AllPoints)
    #
    # plt.ylabel('Count')
    # plt.xlabel('Gray')
    # # plt.xlim(0, 241)
    # # plt.xticks(range(0, 241, 20))
    # plt.show()


    # plt.plot(Std_lst)
    # plt.ylabel('Std')
    # plt.xlabel('Height')
    # plt.xlim(0,241)
    # plt.xticks(range(0, 241, 20))
    # plt.show()

    cv2.line(image, (left_boundle, idx - dist+1), (right_boundle, idx - dist+1), color, thickness)
    cv2.line(image, (left_boundle, idx + dist), (right_boundle, idx + dist), color, thickness)
    cv2.line(image, (left_boundle, idx - dist), (left_boundle, idx + dist), color, thickness)
    cv2.line(image, (right_boundle, idx - dist), (right_boundle, idx + dist), color, thickness)


    # print(idx,idx-dist)
    original_height, original_width = image.shape[:2]
    window_width, window_height = 600, 500
    # 定义窗口的目标大小（比例可以根据需求调整）
    target_width = 600
    target_height = int((target_width / original_width) * original_height)

    #展示图像窗口
    cv2.namedWindow('Adjusted Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Adjusted Window', target_width, target_height)
    cv2.imshow('Adjusted Window', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


funcmain()