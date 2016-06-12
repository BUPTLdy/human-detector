
## 数据集介绍

训练数据来自[INRIA Person Dataset](http://pascal.inrialpes.fr/data/human/),其中正样本为64\*128的人体图像，负样本为64\*128的非人体图像，如下图所示。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-27/56925025.jpg)


## HOG特征


局部目标的外表和形状可以被局部梯度或边缘方向的分布很好的描述,即使我们不知道对应的梯度和边缘
的位置。在实际操作中,将图像分为小的细胞单元 (cells) ,每个细胞单元计算一个梯度方向 ( 或边缘方向 )
直方图。为了对光照和阴影有更好的不变性,需要对直方图进行对比度归一化,可以通过将细胞单元组
成更大的块 (blocks) 并归一化块内的所有细胞单元来实现。我们将归一化的块描述符叫做 HOG 描述子。将
检测窗口中的所有块的 HOG 描述子组合起来就形成了最终的特征向量,然后使用 SVM 分类器进行人体检
测,见下图。
![](http://7xritj.com1.z0.glb.clouddn.com/16-3-31/19766259.jpg)

HOG特征有个优点,它们提取的边缘和梯度特征能很好的抓住局部形状的特点,并且由于是
       在局部进行提取,所以对几何和光学变化都有很好的不变性:变换或旋转对于足够小的区域影响很小。
对于人体检测,在粗糙的空域采样 (coarse spatial sampling) 、精细的方向采样 (fine orientationsampling)
和较强的局部光学归一化 (stronglocal photometric normalization) 这些条件下,只要行人大体上能够保持
直立的姿势,就容许有一些细微的肢体动作,这些细微的动作可以被忽略而不影响检测效果。

###伽马颜色归一化
用不同的幂值 (gamma 参数 ) 评价了几种颜色空间,有灰度空间、 RGB 、 LAB ,结果表明,这些
规范化对结果影响很小,可能是由于随后的描述子归一化能达到相似的效果。

###梯度计算

不同的梯度计算方法对检测器性能有很大影响,但事实证明最简单的梯度算子结果是最好的。采用了最简单的一维离散微分模板算子。测试表明，使用 Sobel 算子等其它算子或是引入高斯平滑反而会
造成性能降低。对于带颜色的图像,分别计算每个颜色通道的梯度,以范数最大者作为该点的梯度向量。

### 空间 / 方向 bin 统计
如下图的一个包含行人的图像，红色框标记一个 8 × 8 单元，这些 8 × 8 的单元将被用来计算 HOG 描述符。



<div style="text-align: center">
<img src="http://7xritj.com1.z0.glb.clouddn.com/16-3-31/94180844.jpg"/>
</div>
在每
个单元中，我们在每个像素上计算梯度矢量，将得到 64 个梯度矢量，梯度矢量相角在 0◦ → 180◦ 之间分布，我们对
相角进行分箱 (bin)，每箱 20◦，一共 9 箱 (Dalal 和 Triggs 得到的最佳参数)。具有某一相角的梯度矢量的幅度按照权
重分配给直方图。这涉及到权重投票表决机制， Dalal 和 Triggs 发现，采用梯度幅度进行分配表现最佳。例如，一个
具有 85 度相角的梯度矢量将其幅度的 1/4 分配给中心为 70◦ 的箱，将剩余的 3/4 幅度分配给中心为 90◦ 的箱。这样
就得到了下面的方向梯度直方图。


<div style="text-align: center">
<img src="http://7xritj.com1.z0.glb.clouddn.com/16-3-31/13507779.jpg"/>
</div>
上面分配幅度的方法可以减少恰好位于两箱边界的梯度矢量的影响，否则，如果一个强梯度矢量恰好在边界上，
其相角的一个很小的绕动都将对直方图造成非常大的影响。同时，在计算出梯度后进行高斯平滑，也可以缓解这种
影响。
另一方面，特征的复杂程度对分类器的影响很大。通过直方图的构造，我们将特征 64 个二元矢量量化为特征 9 个
值，很好地压缩了特征的同时保留了单元的信息。设想对图像加上一些失真，方向梯度直方图的变化也不会很大，这
是 HOG 特征的优点。

###归一化处理

前面提到，对图像所有像素进行加减后梯度矢量不变，接下来引入梯度矢量的标准化，使得其在像素值进行乘法运
算后仍然保持不变。如果对单元内的像素值都乘以某一常数，梯度矢量的幅度明显会发生变化，幅度会增加常数因子,
相角保持不变，这会造成整个直方图的每个箱的幅度增加常数因子。为了解决这个问题，需要引入梯度矢量标准化，
一种简单的标准化方法是将梯度矢量除以其幅度，梯度矢量的幅度将保持 1，但是其相角不会发生变化。引入梯度矢
量标准化以后，直方图各箱幅度在图像像素值整体乘以某个因子 (变化对比度) 时不会发生变化。
除了对每个单元的直方图进行标准化外，另外一种方法是将固定数量的空域邻接的单元封装成区块，然后在区块上
进行标准化。 Dalal 和 Triggs 使用 2 × 2 区块 (50% 重叠)，即 16 × 16 像素。将一个区块内的四个单元的直方图信息
整合为 36 个值的特征 (9 × 4), 然后对这个 36 元矢量进行标准化。 Dalal 和 Triggs 考察了四种不同的区块标准化算
法，设 v 为未标准化的区块梯度矢量， $||v||_{k}(k = 1, 2)$ 是 v 的 k-范数 (norm),e 是一个很小的常数 (具体值并不重要)，
四种标准化算法如下：

<div style="text-align: center">
<img src="http://7xritj.com1.z0.glb.clouddn.com/16-3-31/98414203.jpg"/>
</div>

L2-Hys 是在 L2-norm 后进行截断，然后重新进行标准化。 Dalal 和 Triggs 发现 L2-Hys,L2-norm,L1-sqrt 性能相似，
L1-norm 性能稍有下降，但都相对于未标准化的梯度矢量有明显的性能提升。
区块重叠的影响是使得每个单元会在最终得到的 HOG 描述符中其作用的次数大于 1 次 (角单元出现 1 次，边单元
出现 2 次，其它单元出现 4 次)，但每次出现都在不同的区块进行标准化。定义一个区块位移的步长为 8 像素，则可
以实现 50% 的重叠。
如果检测器窗口为 64x128像素，则会被分为 7 × 15 区块，每个区块包括 2 × 2 个单元，每个单元包括 8 × 8 像素，每个
区块进行 9 箱直方图统计 (36 值)，最后的总特征矢量将有 7 × 15 × 4 × 9 = 3780 个特征值元素。

提取HOG特征的方法使用了skimage库中的hog函数。

```python

def extract_features():
    des_type = 'HOG'

    # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)

    # If feature directories don't exist, create them
    if not os.path.isdir(neg_feat_ph):
        os.makedirs(neg_feat_ph)

    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        #print im_path

        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print "Positive features saved in {}".format(pos_feat_ph)

    print "Calculating the descriptors for the negative samples and saving them"
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd = hog(im,  orientations, pixels_per_cell, cells_per_block, visualize, normalize)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(neg_feat_ph, fd_name)

        joblib.dump(fd, fd_path)
    print "Negative features saved in {}".format(neg_feat_ph)
```

## 训练SVM

因为每张图片提取出来的HOG特征有6480维，所以我们使用线性SVM就足够可分。
```python
def train_svm():
    pos_feat_path = '../data/features/pos'
    neg_feat_path = '../data/features/neg'

    # Classifiers supported
    clf_type = 'LIN_SVM'

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    print np.array(fds).shape,len(labels)
    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print "Training a Linear SVM Classifier"
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        joblib.dump(clf, model_path)
        print "Classifier saved to {}".format(model_path)

```

## 进行人体检测

因为对进行人体检测的输入图片大小是未知的，所以需要对图片进行尺度缩放，使用的方法如下所示：

```python
from skimage.transform import pyramid_gaussian
pyramid_gaussian(im, downscale=downscale)
```

在缩放的尺度上对图片进行滑动窗口检测，可能会在不同尺度上都检测到了目标，这样会造成标记的混乱，可以使用非极大值抑制的方法对重复标记的的目标经行剔除。可以从imutils包中导入非极大值抑制函数。

imutils包安装
```
sudo pip install imutils
```
使用非极大值抑制函数：
```python
from imutils.object_detection import non_max_suppression
```

完整检测代码：

```python
def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0)
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])



def detector(filename):
    im=cv2.imread(filename)
    im = imutils.resize(im, width=min(400, im.shape[1]))
    min_wdw_sz = (64, 128)
    step_size = (10, 10)
    downscale = 1.25
    # 导入SVM模型
    clf = joblib.load(model_path)

    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0
    # 在图像金字塔模型中对每个滑动窗口经行预测
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        # This list contains detections at the current scale
        cd = []
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            # 计算每个窗口的Hog特征
            im_window=color.rgb2gray(im_window)
            fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)

            fd=fd.reshape(1,-1)
            pred = clf.predict(fd)
            if pred == 1:

                if clf.decision_function(fd)>0.5:
                    detections.append((x, y, clf.decision_function(fd),#样本点到超平面的距离
                        int(min_wdw_sz[0]*(downscale**scale)),
                        int(min_wdw_sz[1]*(downscale**scale))))
                    cd.append(detections[-1])

        scale+=1


    clone = im.copy()

    # 画出矩形框
    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 255, 0), thickness=2)

    rects = np.array([[x, y, x + w, y + h] for (x, y,_, w, h) in detections])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.3)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)


    plt.axis("off")
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("Raw Detections before NMS")
    plt.show()

    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()


```

## 效果演示

非极大值抑制处理前：

![](http://7xritj.com1.z0.glb.clouddn.com/16-5-27/45995282.jpg)
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-27/13402395.jpg)

非极大值抑制处理后：

![](http://7xritj.com1.z0.glb.clouddn.com/16-5-27/43202553.jpg)
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-27/41627345.jpg)
