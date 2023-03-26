# 使用PWC-net进行PIV速度场计算



## 问题概述

基于预训练好的PWC-net进行PIV速度场计算，PWC-net来自https://github.com/sniklaus/pytorch-pwc

相较计算机视觉领域的图像，PIV图像一般为单通道灰度图像，在此code中，对图像进行单通道至三通道的转换，并使用原代码和预训练参数进行光流计算。

结果表明，尽管PWC-net未曾在PIV数据集进行训练，PWC-net仍可以有效计算出PIV速度场，并展现出较高的精度和速度，其可靠性还有待进一步的探究。



## 参考链接

https://blog.csdn.net/m_buddy/article/details/118500301

https://blog.csdn.net/u012348774/article/details/112123638

https://blog.csdn.net/qq_43307074/article/details/127338540

https://github.com/sniklaus/pytorch-pwc