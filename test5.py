# coding: utf-8

import cv2
import cv2.cv as cv
import PIL.Image as Image
import StringIO

capture = cv2.VideoCapture(0)
capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv.CV_CAP_PROP_FRAME_COUNT, 100)

while True:
    ret, img = capture.read()   # 从摄像头读取图片
    cv2.imshow('Video', img)    # 显示图片
    key = cv2.waitKey(30) & 0xFF
    if key == 27:   # ESC
        break
    elif key == 32:  # Space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图片从BGR转为RGB
        image = Image.fromarray(img)    # 将图片转为Image对象
        buf = StringIO.StringIO()   # 生成StringIO对象
        image.save(buf, format="JPEG")  # 将图片以JPEG格式写入StringIO
        jpeg = buf.getvalue()   # 获取JPEG图片内容
        buf.close()     # 关闭StringIO
        print '文件大小：', len(jpeg)/1024, 'KB'
        f = open('a.jpg', 'w')
        f.write(jpeg)
        f.close()

capture.release()
