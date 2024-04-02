from PIL import Image
import os
def func1(file):
    def resize_image(image, target_height):
        # 计算调整后的宽度，保持原始宽高比例
        width_percent = (target_height / float(image.size[1]))
        target_width = int((float(image.size[0]) * float(width_percent)))

        # 调整图片大小
        resized_image = image.resize((target_width, target_height), Image.LANCZOS)

        return resized_image
    # 打开两张图片


    #image2 = Image.open('picture2/'+file)
    file1=file.split('.')[0]
    image2 = Image.open('picture2_copy/'+file.split('.')[0]+'.png')
    image1 = Image.open('picture2/' + file)
    #image2 = Image.open('picture2_copy/2023_09_12 04-45-42-857.png')


    # 获取图片的大小
    width1, height1 = image1.size
    width2, height2 = image2.size

    target_height = 500

    # 调整图片大小
    resized_image1 = resize_image(image1, target_height)
    resized_image2 = resize_image(image2, target_height)

    # 获取调整后图片的大小
    width1, height1 = resized_image1.size
    width2, height2 = resized_image2.size

    # 计算合并后的图片大小
    new_width = width1 + width2
    new_height = max(height1, height2)

    # 创建一个新的空白图片
    new_image = Image.new('RGB', (new_width, new_height))

    # 将调整后的图片粘贴到新图片上
    new_image.paste(resized_image1, (0, 0))
    new_image.paste(resized_image2, (width1, 0))

    # 保存合并后的图片
    new_image.save('picture22/'+file.split('.')[0]+'.jpg')

    # 显示合并后的图片（可选）
    #new_image.show()

#func1('2023_09_12 04-45-42-857.bmp')
def get_file_names(directory_path):
    file_names = []

    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        # 检查是否是文件而不是子目录
        if os.path.isfile(os.path.join(directory_path, filename)):
            file_names.append(filename)

    return file_names

directory_path = 'picture2'
# 调用函数获取文件名列表
file_names_list = get_file_names(directory_path)

for file in file_names_list:
    func1(file)