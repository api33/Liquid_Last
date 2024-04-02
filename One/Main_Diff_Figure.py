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
    return  (int(midpoint_x), int(midpoint_y))


def funcmain():
    # 读取图像
    image_path = 'picture/2023_10_24 21-23-19-439.bmp'  # 替换成你的图像路径
    #image_path = 'picture/2023_10_24 21-23-20-296.bmp'  # 替换成你的图像路径
    #image_path = 'picture/20221217165719077.bmp'  # 替换成你的图像路径
    #image_path = 'picture/20221217165722270.bmp'  # 替换成你的图像路径
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_copy=image.copy()
    # 获取图像宽度和高度
    height, width = image.shape


    # 选择一个确定的高度
    chosen_height = 860 # 替换成你实际需要的高度值


    gray_2000=[int(image[h,2000]) for h in range(height)]
    #print(gray_2000)

    differences=[]
    # 计算相邻数据点的差异
    for i in range(len(gray_2000)):
        start=max(0,i-10)
        differences.append(gray_2000[i] - gray_2000[start])
    idx=differences.index(min(differences))
    gray_values = [image[chosen_height, x] for x in range(width)]

    # 计算每个像素点距离最左侧的距离
    distances = list(range(width))

    # 绘制不同高度的灰度变化曲线
    # plt.plot(gray_2000)
    # plt.ylabel('Gray')
    # plt.xlabel('Height')
    # plt.show()

    #绘制差异图
    # plt.plot(differences)
    # plt.xlabel('height')
    # plt.ylabel('Difference')
    # plt.show()



    color = (0, 0, 255)  # 线的颜色，这里使用RGB格式 (0, 255, 0) 表示绿色
    thickness = 2  # 线的宽度
    dist=120
    left_boundle=1750
    right_boundle=2250
    cv2.line(image, (left_boundle, idx - dist), (right_boundle, idx - dist), color, thickness)
    cv2.line(image, (left_boundle, idx + dist), (right_boundle, idx + dist), color, thickness)
    cv2.line(image, (left_boundle, idx - dist), (left_boundle, idx + dist), color, thickness)
    cv2.line(image, (right_boundle, idx - dist), (right_boundle, idx + dist), color, thickness)
    Rectongle_Points=[]
    Points_lines=[[] for _ in range(2*dist+1)]
    x_points=[]
    y_points=[]
    Differences_Lines=[[] for _ in range(2*dist+1)]
    count=0
    for h in range(idx-dist,idx+dist+1):
        for w in range(left_boundle,right_boundle+1):
            if count<20:
                #print(h)
                count+=1
            x_points.append(h-(idx-dist))
            y_points.append(image_copy[h, w])
            Rectongle_Points.append((h-(idx-dist), image_copy[h, w]))
            Points_lines[h-(idx-dist)].append(image_copy[h,w])
            if w<right_boundle:
                dif=int(image_copy[h,w+1])-int(image_copy[h,w])
                Differences_Lines[h-(idx-dist)].append(dif)
    Average_Diff=[]
    for v in Differences_Lines:
        Average_Diff.append(sum(v)//len(v))

    # plt.plot(Average_Diff)
    # plt.ylabel('Average')
    # plt.xlabel('Height')
    # plt.show()


    temp1=[[] for _ in range(len(Points_lines))]
    for i in range(len(Points_lines)):
        lst = Points_lines[i]
        for j in range(len(lst)-10):
            temp1[i].append(abs(int(lst[j+10])-int(lst[j])))
    average1=[]
    for v in temp1:
        average1.append(sum(v)/len(v))
    print(average1)
    print(Points_lines[0])
    # print(sum(temp1)/len(temp1))
    plt.plot(average1)
    plt.ylabel('Average')
    plt.xlabel('Height')
    plt.show()

    # plt.plot([abs(int(Points_lines[0][i+1])-int(Points_lines[0][i])) for i in range(len(Points_lines[0])-1)])
    # plt.ylabel('Average')
    # plt.xlabel('Height')
    # plt.show()

    # plt.plot(Points_lines[0])
    # plt.ylabel('Average')
    # plt.xlabel('Height')
    # plt.show()
    #
    # plt.plot(Points_lines[dist])
    # plt.ylabel('Average')
    # plt.xlabel('Height')
    # plt.show()

    temp2=[]
    for i in range(len(Points_lines[dist])-30):
        temp2.append(int(Points_lines[dist][i+30])-int(Points_lines[dist][i]))
    print("sum",sum(temp2)/len(temp2))
    print(Points_lines[dist])

    average_lines=[sum(v)//len(v) for v in Points_lines]
    points=[(i,v) for i,v in enumerate(average_lines)]
    midpoint_x,midpoint_y=generate_approximate_curve(points)
    count1=0
    count2=0
    other_points_x1=[]
    other_points_y1=[]
    other_points_x2=[]
    other_points_y2=[]
    for x,y in Rectongle_Points:
        if x<midpoint_x and y<midpoint_y:
            count1+=1
            other_points_x1.append(x)
            other_points_y1.append(y)
        if x>midpoint_x and y>midpoint_y:
            other_points_x2.append(x)
            other_points_y2.append(y)
            count2+=1




    print(count1,count2,len(Rectongle_Points))
    plt.plot(average_lines)
    plt.ylabel('Average')
    plt.xlabel('Height')
    print(len(Points_lines))
    # 绘制矩形的两个角点
    plt.scatter(x_points, y_points, color='red', label='Points')
    plt.scatter(other_points_x1, other_points_y1, color='green', label='Points')
    plt.scatter(other_points_x2, other_points_y2, color='blue', label='Points')
    plt.plot([0, 225], [midpoint_y,midpoint_y], marker='o')
    plt.plot([midpoint_x,midpoint_x], [50,200], marker='o')
    #plt.show()

    # window_width, window_height = 1000, 800
    # # #展示图像窗口
    # cv2.namedWindow('Adjusted Window', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Adjusted Window', window_width, window_height)
    # cv2.imshow('Adjusted Window', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


funcmain()