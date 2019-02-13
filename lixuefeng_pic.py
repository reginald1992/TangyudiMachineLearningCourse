#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : lixuefeng_pic.py
# @Author: Shulin Liu
# @Date  : 2019/2/13
# @Desc  :
from PIL import Image, ImageFilter, ImageOps

# 先到相应的路径下加载到这张图片
img = Image.open('lixuefeng_dog2.jpg')  # jpg和png格式均可


def Formula(a, b, alpha):
    return min(int(a * 255 / (256 - b * alpha)), 255)

# 通过双层for循环将图片转换


def lixuefeng(img, blur=25, alpha=1.0):
    img1 = img.convert('L')  # 图片转换成灰色
    img2 = img1.copy()
    img2 = ImageOps.invert(img2)
    for i in range(blur):  # 模糊度
        img2 = img2.filter(ImageFilter.BLUR)
    width, height = img1.size
    for x in range(width):
        for y in range(height):
            a = img1.getpixel((x, y))
            b = img2.getpixel((x, y))
            img1.putpixel((x, y), Formula(a, b, alpha))
    img1.show()  # 展示图片效果


if __name__ == '__main__':
    lixuefeng(img)
