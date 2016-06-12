

## 数据集介绍

训练数据来自[INRIA Person Dataset](http://pascal.inrialpes.fr/data/human/),其中正样本为64\*128的人体图像，负样本为64\*128的非人体图像，如下图所示。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-27/56925025.jpg)


## HOG特征

HOG特征详细介绍：[HOG论文笔记](http://buptldy.github.io/2016/03/31/2016-03-31-HOG%E8%AE%BA%E6%96%87%E6%80%BB%E7%BB%93/)提取HOG特征的方法使用了skimage库中的hog函数。

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
