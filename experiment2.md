实验二 图像增强实验报告
姓名：文天翔    学号：202410315037
班级：人工智能242    实验日期：2026年10月8日
一 实验目的
掌握 OpenCV 的图像读取、BGR 与 RGB 转换及灰度转换方法；观察椒盐噪声和高斯噪声的特点；比较均值、中值和高斯滤波的去噪效果，并通过手工实现彩色中值滤波理解邻域运算和边界处理。
二 实验内容
2.1 实验环境与依赖
在 Windows 本机的 Anaconda cv 虚拟环境中运行实验。Python 3.11.17，OpenCV 4.12.0，NumPy 2.2.6，Matplotlib 3.11.2，scikit-image 0.26.0。依赖检查 python -m pip check 通过。

图1 本机 Python 环境与依赖检查
import cv2
from skimage.util import random_noise
import numpy as np
from matplotlib import pyplot as plt
本实验固定随机种子 20261008 与 20261009，滤波核统一取 5×5。结果图像、运行日志和定量数据均保存到实验2目录，便于复查。


2.2 读取图像与颜色空间转换
实验输入 p1.jpg 来自参考文档第2页的 RGB 猫图：提取嵌入图像，裁去标题与坐标轴，再缩放为 640×640。该输入经过文档压缩和缩放，因此本次像素数值与参考示例略有差异。
使用 OpenCV 读取图像，得到形状 (640, 640, 3)、类型 uint8 的 BGR 数组。考虑到实验目录含中文，程序采用 np.fromfile 与 cv2.imdecode 读取文件；解码后的通道顺序与 cv2.imread 相同。
img = cv2.imdecode(np.fromfile(str(ROOT/'p1.jpg'),
                   dtype=np.uint8), cv2.IMREAD_COLOR)
(b, g, r) = img[100, 100]
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

图2 实际绘图窗口中的颜色空间转换结果
像素与通道顺序
采样位置	BGR 数值	RGB 数值	灰度值
第100行 第100列	(84, 86, 241)	(241, 86, 84)	132

Matplotlib 将三通道数组解释为 RGB。若直接传入 OpenCV 的 BGR 数据，红蓝通道显示颠倒，猫的眼睛呈蓝色；转换后眼睛恢复黄色、衣服恢复蓝色。cvtColor 调整的是数据通道顺序，转换后才符合 Matplotlib 的显示约定。
灰度转换把三通道映射为单通道亮度，保留轮廓和明暗关系。红、绿、蓝通道的亮度权重分别约为 0.299、0.587、0.114，本次采样点的灰度值为 132。


2.3 添加椒盐噪声与高斯噪声
以正确的 RGB 图像为输入，使用 scikit-image 的 random_noise 生成两类噪声。返回值是 [0, 1] 范围的浮点数组；通过明确指定随机数生成器，可以复现同一组噪声。
sp_noise_img = random_noise(
    rgb_img, mode='s&p', amount=0.4,
    rng=np.random.default_rng(20261008))
gus_noise_img = random_noise(
    rgb_img, mode='gaussian', mean=0.2, var=0.03,
    rng=np.random.default_rng(20261009))

图3 原图与两类噪声的实际显示结果
噪声参数与观察
噪声类型	本次参数	可见变化
椒盐噪声	amount=0.4	大量离散噪点
高斯噪声	mean=0.2  var=0.03	随机颗粒与整体变亮

椒盐噪声将部分数组元素置为 0 或 1，产生突出的暗点、亮点。对本次三通道彩色输入，替换可分别发生在不同通道，因此除了黑白点还会出现彩色点；amount=0.4 描述的是元素替换概率，不能直接解释为恰好40%的完整像素被替换。
高斯噪声向像素添加服从正态分布的扰动，方差开平方所得的标准差约为 0.173。由于均值设为 0.2 而非 0，噪声不仅增加随机颗粒，也使图像整体变亮；输出在 [0, 1] 范围截断，亮部出现饱和。


2.4 三种滤波方法比较
对两类噪声分别执行 5×5 均值、中值和高斯滤波，保持同一类噪声输入相同。均值滤波对浮点数组求邻域平均；中值与高斯滤波按参考流程先转换为 uint8，避免中值滤波的数据类型限制。
mean_sp = cv2.blur(sp_noise_img, (5, 5))
mean_gus = cv2.blur(gus_noise_img, (5, 5))
sp_uint = (sp_noise_img * 255).astype(np.uint8)
gus_uint = (gus_noise_img * 255).astype(np.uint8)
mid_sp = cv2.medianBlur(sp_uint, 5)
mid_gus = cv2.medianBlur(gus_uint, 5)
gauss_sp = cv2.GaussianBlur(sp_uint, (5, 5), 0)
gauss_gus = cv2.GaussianBlur(gus_uint, (5, 5), 0)

图4 上排为椒盐噪声 下排为高斯噪声的滤波结果
均值滤波以邻域平均代替中心像素，能够减弱随机波动，但会使细线和边缘模糊。中值滤波选择邻域中间值，对极端噪声的影响较小，猫的眼睛轮廓和背景在椒盐噪声去除后较清晰。高斯滤波给中心邻域较高权重，平滑效果较自然，但对强椒盐噪点的抑制不如中值滤波。


2.5 手工实现彩色中值滤波
逐个处理 R、G、B 通道，对边缘做复制填充，再遍历每个像素并计算 5×5 邻域中值。该边界处理与 OpenCV 中值滤波采用的复制边界一致，可直接核对整幅图像。
def manual_median_filter_color(image, kernel_size=5):
    pad = kernel_size // 2
    filtered_img = np.zeros_like(image)
    for c in range(3):
        channel = image[:, :, c]
        padded_channel = np.pad(channel, pad, mode="edge")
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                region = padded_channel[
                    i:i+kernel_size, j:j+kernel_size]
                filtered_img[i, j, c] = np.median(region)
    return filtered_img

manual_mid = manual_median_filter_color(sp_uint, 5)
diff = cv2.absdiff(manual_mid, mid_sp)
assert diff.max() == 0

图5 椒盐噪声输入 手工中值滤波与 OpenCV 中值滤波
本次手工滤波耗时 7.591 秒，OpenCV 中值滤波耗时 0.001622 秒。逐个比较 640×640×3 = 1,228,800 个通道数值，最大绝对差值为 0，不同数值个数为 0，证明当前输入与参数下两种实现完全一致。耗时为本次单次运行测量，不能直接视为稳定性能基准。
两种实现都明显减少了离散噪点，并保留主体颜色与轮廓。手工逐像素循环便于理解算法，但执行速度明显低于 OpenCV 的优化实现。


三 实验结果与分析
3.1 定量结果
以未加噪的 RGB 输入图像为参考，把参与比较的数据归一化到 [0, 1]，在全图三个通道上计算均方误差 MSE 和峰值信噪比 PSNR。MSE 越小、PSNR 越高，表示与参考图像越接近。
MSE 为全图通道平方误差的平均值；归一化图像的 PSNR 按下式计算，单位为 dB。下表数值由本机运行程序直接导出，保留三位小数或六位小数。
PSNR = 10log10(1MSE)
噪声与处理方法	MSE	PSNR dB
椒盐噪声 未滤波	0.143252	8.439
椒盐噪声 均值滤波	0.024020	16.194
椒盐噪声 中值滤波	0.001868	27.287
椒盐噪声 高斯滤波	0.027645	15.584
椒盐噪声 手工中值滤波	0.001868	27.287
高斯噪声 未滤波	0.048995	13.099
高斯噪声 均值滤波	0.031003	15.086
高斯噪声 中值滤波	0.030995	15.087
高斯噪声 高斯滤波	0.030385	15.173

3.2 滤波效果分析
椒盐噪声下，中值滤波的 PSNR 为 27.287 dB，明显高于均值滤波的 16.194 dB 与高斯滤波的 15.584 dB。邻域中值受到极端值影响较小，能较有效地去除离散脉冲噪声；均值与高斯滤波会把噪点影响扩散到邻域，同时模糊图像细节。手工中值滤波的各项指标与 OpenCV 完全相同。
本次高斯噪声下，高斯滤波的 PSNR 为 15.173 dB，略高于均值滤波的 15.086 dB 和中值滤波的 15.087 dB。三者差距很小，均值与中值的结果尤其接近。此结论仅针对当前图像、固定噪声样本与 5×5 滤波核，不能推广为高斯滤波在所有高斯噪声条件下必然最优。
3.3 非零均值噪声的影响
添加高斯噪声后，全图平均误差约为 +0.149；高斯滤波后仍约为 +0.147。滤波主要减弱随机波动，无法消除 mean=0.2 带来的整体亮度偏移；同时截断会造成亮部信息损失，因此图像虽更平滑，却不能完全恢复原始亮度。


3.4 本机运行验证记录
运行日志记录了实际软件版本、采样点数值、噪声参数、滤波指标和手工实现的一致性检查。以下为程序完成后自行截取的 PowerShell 窗口。

图6 本机完整程序运行与一致性检查结果
四 实验小结
本次实验完成了彩色图像读取、颜色空间转换、两类噪声添加和三种滤波方法比较，并手工实现了彩色中值滤波。实验表明，中值滤波适合当前的强椒盐噪声；均值和高斯滤波可以平滑随机噪声，但需同时考虑边缘模糊和亮度偏移。
实现过程中需要正确区分 BGR 与 RGB，处理好浮点图像和 uint8 图像之间的范围转换，并保持边界规则一致。手工滤波与 OpenCV 的逐像素比较验证了算法实现，定量指标则补充了仅凭视觉观察难以判断的差异。
实验文件与复现方法
完整源代码保存在“图像增强实验.py”；双击“运行实验.cmd”可在 cv 环境中重新运行并自动截图。“截图”目录保存6张实际窗口截图，“output_images”保存处理图像，“过程记录”保存运行日志、输入来源、实验结果 JSON 和定量对比 CSV。
参考资料：实验二参考文档，实验目的、步骤及报告实例，第1至13页。
