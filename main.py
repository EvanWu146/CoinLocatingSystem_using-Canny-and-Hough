import numpy as np
from PIL import Image
import cv2

def get_img(path):  # 从给定路径读取图片，并转为灰度图
    print("Getting image...")
    image = Image.open(path)
    raw_img = np.copy(image)  # 读取图片至numpy数组中
    img = raw_img.mean(axis=2)  # 将RGB图片转为灰度图（对RGB三个通道取均值）
    print("Over.")
    return img, raw_img


def get_gaussian_kernel(k, sigma):  # 获取高斯滤波器
    print("Getting gaussian filter...")
    kernel_h = np.zeros((2 * k + 1, 2 * k + 1))
    for i in range(0, 2 * k + 1):
        for j in range(0, 2 * k + 1):
            X, Y = i - k, j - k
            kernel_h[i][j] = np.exp(- (X ** 2 + Y ** 2) / (2 * sigma ** 2))
    kernel_h /= np.sum(kernel_h)  # 归一化
    print("Over.")
    return kernel_h

def conv2d(img, kernel, pad=(0, 0), stride=1):  # 二维卷积（只接收2维np数组）
    print("Start to convolution...")
    H, W = img.shape
    kernel_h, kernel_w = kernel.shape

    out_h = (H + 2 * pad[0] - kernel_h) // stride + 1
    out_w = (W + 2 * pad[1] - kernel_w) // stride + 1

    new_img = np.pad(img, [[pad[0], pad[0]], [pad[1], pad[1]]], 'constant', constant_values=(0, 0))
    col = np.zeros((kernel_h, kernel_w, out_h, out_w))

    for y in range(kernel_h):  # im2col方法，将卷积图像按照卷积核的框展开成列
        for x in range(kernel_w):
            col[y, x, :, :] = new_img[y:(y + stride * out_h):stride, x:(x + stride * out_w):stride]

    new_img = col.transpose((2, 3, 0, 1)).reshape(out_h * out_w, -1)
    kernel = kernel.reshape((1, -1)).T  # 展开卷积核，并转置
    print("Over.")
    return np.dot(new_img, kernel).T.reshape((out_h, out_w))  # 返回乘法矩阵运算

def get_gradient_pic(img):  # 基于sober算子的边缘提取
    print("Begin to calculating the gradient...")
    sober_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sober_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    result_x = conv2d(img, sober_x)
    result_y = conv2d(img, sober_y)

    print("Over.")
    return result_x, result_y

def NMS(Gx, Gy):  # 非极大值抑制
    print("Processing the Non-Maximum Suppression...")
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    Gx[np.abs(Gx) <= 1e-5] = 1e-5  # 防止分母为0

    # 计算梯度的方向
    temp = np.arctan(Gy, Gx) / np.pi * 180.
    temp[temp < -22.5] += 180.
    angle = np.zeros_like(temp, dtype=np.uint8)
    # 以22.5度为分界线，将角度取整至45度的倍数
    angle[np.where(temp <= 22.5)] = 0
    angle[np.where((temp > 22.5) & (temp <= 67.5))] = 45
    angle[np.where((temp > 67.5) & (temp <= 112.5))] = 90
    angle[np.where((temp > 112.5) & (temp <= 157.5))] = 135

    H, W = angle.shape
    di0, dj0, di1, dj1 = 0, 0, 0, 0
    for i in range(H):
        for j in range(W):
            # 根据角度矩阵，计算出偏移坐标量，以进行当前梯度的正负方向比较
            if angle[i, j] == 0:
                di0, di1, dj0, dj1 = -1, 0, 1, 0
            elif angle[i, j] == 45:
                di0, di1, dj0, dj1 = -1, 1, 1, -1
            elif angle[i, j] == 90:
                di0, di1, dj0, dj1 = 0, -1, 0, 1
            elif angle[i, j] == 135:
                di0, di1, dj0, dj1 = -1, -1, 1, 1

            # 对于边缘像素的处理
            if j == 0:
                di0 = max(di0, 0)
                dj0 = max(dj0, 0)
            if j == W - 1:
                di0 = min(di0, 0)
                dj0 = min(dj0, 0)
            if i == 0:
                di1 = max(di1, 0)
                dj1 = max(dj1, 0)
            if i == H - 1:
                di1 = min(di1, 0)
                dj1 = min(dj1, 0)

            # 如果当前像素的梯度强度与另外两个像素相比不是最大的，则该像素点将被抑制
            if max(max(G[i, j], G[i + di1, j + di0]), G[i + dj1, j + dj0]) != G[i, j]:
                G[i, j] = 0
    print("Over.")
    return G

def DTD(src, HT=100, LT=10):  # 双阈值检测
    print("Processing Double Threshold Detection...")
    H, W = src.shape

    # 高低阈值的初始化
    src[src >= HT] = 255
    src[src < LT] = 0

    src = np.pad(src,[[1, 1], [1, 1]], 'constant', constant_values=(0, 0))  # 对原图像进行填充
    n8 = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)  # 8邻域像素比较法

    for i in range(1, H + 2):
        for j in range(1, W + 2):
            if LT <= src[i,j] <= HT:  # 针对在阈值中间的像素，如果其邻接着强边，则将其设置为强边
                if np.max(src[i-1:i+2, j-1:j+2] * n8) >= HT:
                    src[i, j] = 255
                else:  # 否则被抑制
                    src[i, j] = 0

    print("Over.")
    return src[1:H+1, 1:W+1]  # 取消padding

def show_img(img):
    new_img = Image.fromarray(img)
    new_img.show()


if __name__ == '__main__':
    "----------以下为Canny算法，检测硬币的边缘----------"
    img, raw = get_img('picture.jpg')
    kernel = get_gaussian_kernel(k=1, sigma=1.4)
    img = conv2d(img, kernel, pad=(1, 1))
    Gx, Gy = get_gradient_pic(img)
    G = NMS(Gx, Gy)
    G = DTD(src=G, HT=50, LT=5)
    show_img(G)
    print()

    "----------以下为Hough算法，寻找硬币的圆心坐标和半径----------"
    hough = cv2.HoughCircles(np.uint8(G), cv2.HOUGH_GRADIENT, 1, 400, param1=100, param2=40, minRadius=50, maxRadius=200)
    for circle in hough[0]:
        x, y, r = circle
        cv2.circle(raw, (int(x), int(y)), int(r), (0, 0, 255), 3)
        cv2.circle(raw, (int(x), int(y)), 5, (0, 255, 0), -1)
        print("The center of the circle: ", (int(x), int(y)), ", the radius: ", int(r))

    cv2.imwrite("result.jpg", raw)




