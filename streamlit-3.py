import os
import streamlit as st
import cv2,numpy as np
import torchvision
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import datasets
import torchvision.transforms as transforms
from experimentmodel import *
from translate import Translator
import pickle
from skimage import feature as ft, color
from joblib import load
from skimage.transform import resize
from skimage.feature import hog
from sklearn.feature_selection import SelectKBest,chi2
st.balloons()
st.snow()
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict
st.subheader("姓名：龙海浪"+"                  "+"学号：2109120127")
st.title(':blue[这是一个自由选择模型的图像识别streamlit应用！] :sunglasses:')

st.write('首先声明该应用的模型识别所用的训练集为 CIFAR-100 数据集')

st.subheader('请您自由选择对图像所提取的特征和使用的模型：')

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

col1, col2 ,= st.columns(2)
optionone,optiontwo,optionthree="","",""
with col1:
    on = st.toggle("不提取特征，采用神经网络",key='disabled')
    if on:
        option1 = st.selectbox(
            "选择神经网络模型",
            ("NET", "MLP", "RESNET18"),
        )
        optionone=option1
with col2:
    option2 = st.selectbox(
        "选择对图像提取的特征",
        ("LBP", "HOG", "SIFT"),
        label_visibility='visible',
        disabled=st.session_state.disabled,
    )
    optiontwo = option2
    option3 = st.selectbox(
        "选择经典模型",
        ("朴素贝叶斯模型", "KNN", "逻辑回归"),
        label_visibility='visible',
        disabled=st.session_state.disabled,
    )
    optionthree=option3
# 上传图片并展示
uploaded_file = st.file_uploader("上传一张图片", type="jpg")

if uploaded_file is not None:
    # 将传入的文件转为Opencv格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    # 展示图片
    st.image(opencv_image, channels="BGR")

if st.button("开始预测",type="primary"):
    if optionone == "NET":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_data = datasets.CIFAR100('./100', train=False, download=True, transform=transform)
    # 加载保存的模型
        model = Net()
        model.load_state_dict(torch.load('models/model_cifar.pt'))

    # 判断是否有GPU
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
            model.cuda()

    # 将模型参数转换为与设备匹配的类型
        if train_on_gpu:
            model = model.cuda()
        else:
            model = model.cpu()
    # 对图片进行转换
        image = Image.open(uploaded_file).convert('RGB')
        image_tensor = transform(image).unsqueeze_(0)

    # 判断是否有GPU
        if train_on_gpu:
            image_tensor = image_tensor.cuda()

    # 进行预测
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    # 获取预测的类别名称
        predicted_class = test_data.classes[predicted[0].item()]

        a = Translator(from_lang="English", to_lang="Chinese").translate(predicted_class)
    # 打印预测结果
        st.write('（NET模型)您上传的图片预测类别结果为：',a,predicted_class)
    elif optionone=="MLP":
        model = MLP(3072, 512, 100)
        # 加载模型权重
        model.load_state_dict(torch.load('models/model_cifar_2.pt'))
        # 定义预处理转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        data_test = datasets.CIFAR100(root="./100", transform=transform, train=False, download=True)
        class_labels = data_test.classes
        # 加载待预测的图片
            # 保存上传的文件到本地
        with open("temp_image.jpg", "wb") as file:
            file.write(uploaded_file.getvalue())
            # 加载保存的文件
        image = cv2.imread("temp_image.jpg")
        # 调整图片大小或进行裁剪
        resized_image = cv2.resize(image, (32, 32))
        # 进行预处理转换
        test_image = transform(resized_image)
        test_image = test_image.flatten()  # 展平图片
        # 增加一维作为 batch 维度
        test_image = test_image.unsqueeze(0)
        # 对图片进行预测
        output = model(test_image)
        pred = output.argmax(1)
        class_name = class_labels[pred.item()]
        a = Translator(from_lang="English", to_lang="Chinese").translate(class_name)
        st.write('(MLP模型）您上传的图片预测类别结果为：', a, class_name)
        # 删除临时图片文件
        os.remove("temp_image.jpg")
    elif optionone == "RESNET18":
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        # 加载保存的模型文件
        model = torch.load("models/CIFAR100_epoch50.pth")
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 调整图像大小为32x32
            transforms.ToTensor(),
            transforms.Normalize(mean, std)  # 标准化图像数据
        ])
        train_data = torchvision.datasets.CIFAR100('./100', train=True, transform=transform, download=True)
        # 要预测的图像文件

        # 加载和预处理图像
        image = Image.open(uploaded_file).convert('RGB')
        image_tensor = transform(image).unsqueeze_(0)

        # 如果有GPU可用，则将图像数据移动到GPU设备
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        # 执行预测
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)

        # 获取预测的类别名称
        class_names = train_data.classes
        predicted_class = class_names[predicted.item()]
        a = Translator(from_lang="English", to_lang="Chinese").translate(predicted_class)
        st.write('（RESNET18模型）您上传的图片预测类别结果为：', a, predicted_class)
    elif optiontwo=="LBP" and optionthree=="朴素贝叶斯模型":
        with open("temp_image.jpg", "wb") as file:
            file.write(uploaded_file.getvalue())
            # 加载保存的文件
        image = cv2.imread("temp_image.jpg")
    # 进行图像裁剪和灰度化
        image = image[2:30, 2:30, :]
        gray_image = color.rgb2gray(image)
    # 进行LBP特征提取
        radius = 2
        n_points = radius * 8
        lbp = ft.local_binary_pattern(gray_image, n_points, radius, method='uniform')
    # 将LBP特征向量转换为一维数组
        lbp_vector = lbp.flatten().astype(float)

    # 加载PCA模型
        pca = load('models/pca_modelLBP.pth')

    # 对特征向量进行PCA降维
        lbp_vector = pca.transform(lbp_vector.reshape(1, -1))

    # 加载朴素贝叶斯模型
        model_beiyensi = load('models/beiyesimodelLBP.pth')
        pred_beiyensi = model_beiyensi.predict(lbp_vector)
        cifar100_labels = unpickle("cifar-100-python/meta")['fine_label_names']
        a = Translator(from_lang="English", to_lang="Chinese").translate(cifar100_labels[pred_beiyensi[0]])
        st.write("LBP特征朴素贝叶斯预测类别：", cifar100_labels[pred_beiyensi[0]], a)
        os.remove("temp_image.jpg")
    elif optiontwo=="LBP" and optionthree=="KNN":
        with open("temp_image.jpg", "wb") as file:
            file.write(uploaded_file.getvalue())
            # 加载保存的文件
        image = cv2.imread("temp_image.jpg")
    # 进行图像裁剪和灰度化
        image = image[2:30, 2:30, :]
        gray_image = color.rgb2gray(image)
    # 进行LBP特征提取
        radius = 2
        n_points = radius * 8
        lbp = ft.local_binary_pattern(gray_image, n_points, radius, method='uniform')
    # 将LBP特征向量转换为一维数组
        lbp_vector = lbp.flatten().astype(float)

    # 加载PCA模型
        pca = load('models/pca_modelLBP.pth')

    # 对特征向量进行PCA降维
        lbp_vector = pca.transform(lbp_vector.reshape(1, -1))
        model_knn = load('models/KNNmodelLBP.pth')
        pred_knn = model_knn.predict(lbp_vector)
        cifar100_labels = unpickle("cifar-100-python/meta")['fine_label_names']
        b = Translator(from_lang="English", to_lang="Chinese").translate(cifar100_labels[pred_knn[0]])
        st.write("LBP特征KNN预测类别：", cifar100_labels[pred_knn[0]],b)
        os.remove("temp_image.jpg")
    elif optiontwo == "LBP" and optionthree == "逻辑回归":
        with open("temp_image.jpg", "wb") as file:
            file.write(uploaded_file.getvalue())
            # 加载保存的文件
        image = cv2.imread("temp_image.jpg")
        # 进行图像裁剪和灰度化
        image = image[2:30, 2:30, :]
        gray_image = color.rgb2gray(image)
        # 进行LBP特征提取
        radius = 2
        n_points = radius * 8
        lbp = ft.local_binary_pattern(gray_image, n_points, radius, method='uniform')
        # 将LBP特征向量转换为一维数组
        lbp_vector = lbp.flatten().astype(float)

        # 加载PCA模型
        pca = load('models/pca_modelLBP.pth')

        # 对特征向量进行PCA降维
        lbp_vector = pca.transform(lbp_vector.reshape(1, -1))
        model_luoji = load('models/luojimodelLBP.pth')
        pred_luoji = model_luoji.predict(lbp_vector)
        cifar100_labels = unpickle("cifar-100-python/meta")['fine_label_names']
        c = Translator(from_lang="English", to_lang="Chinese").translate(cifar100_labels[pred_luoji[0]])
        st.write("LBP特征逻辑回归预测类别：", cifar100_labels[pred_luoji[0]],c)
        os.remove("temp_image.jpg")
    elif optiontwo == "HOG" and optionthree == "朴素贝叶斯模型":
        image = np.array(Image.open(uploaded_file).convert('RGB'))
        resized_image = resize(image, (32, 32))
        # 将图像转换为灰度图像
        image_gray = np.dot(resized_image[..., :3], [0.299, 0.587, 0.114])
        # 计算HOG特征
        orientations = 9
        pixels_per_cell = (8, 8)
        hog_features = hog(image_gray, orientations=orientations, pixels_per_cell=pixels_per_cell)
        model_beiyesi = load('models/beiyesimodelHOG.pth')
        hog_features = hog_features.reshape(1, -1)
        feature_selector = SelectKBest(chi2, k=50)
        hog_features_new = feature_selector.fit_transform(hog_features, [0])  # 使用 [0] 作为目标值示例

        # Predict with each model
        prediction_beiyesi = model_beiyesi.predict(hog_features_new)
        cifar100_labels = unpickle("cifar-100-python/meta")['fine_label_names']
        a = Translator(from_lang="English", to_lang="Chinese").translate(cifar100_labels[prediction_beiyesi[0]])
        st.write("HOG特征朴素贝叶斯模型预测类别为：", prediction_beiyesi[0],a)
    elif optiontwo == "HOG" and optionthree == "KNN":
        image = np.array(Image.open(uploaded_file).convert('RGB'))
        resized_image = resize(image, (32, 32))
        # 将图像转换为灰度图像
        image_gray = np.dot(resized_image[..., :3], [0.299, 0.587, 0.114])
        # 计算HOG特征
        orientations = 9
        pixels_per_cell = (8, 8)
        hog_features = hog(image_gray, orientations=orientations, pixels_per_cell=pixels_per_cell)
        model_KNN = load('models/KNNmodelHOG.pth')
        hog_features = hog_features.reshape(1, -1)
        feature_selector = SelectKBest(chi2, k=50)
        hog_features_new = feature_selector.fit_transform(hog_features, [0])  # 使用 [0] 作为目标值示例

        # Predict with each model
        prediction_KNN = model_KNN.predict(hog_features_new)
        cifar100_labels = unpickle("cifar-100-python/meta")['fine_label_names']
        b = Translator(from_lang="English", to_lang="Chinese").translate(cifar100_labels[prediction_KNN[0]])
        st.write("HOG特征的KNN模型预测类别为：", prediction_KNN[0],b)
    elif optiontwo == "HOG" and optionthree == "逻辑回归":
        image = np.array(Image.open(uploaded_file).convert('RGB'))
        resized_image = resize(image, (32, 32))
        # 将图像转换为灰度图像
        image_gray = np.dot(resized_image[..., :3], [0.299, 0.587, 0.114])
        # 计算HOG特征
        orientations = 9
        pixels_per_cell = (8, 8)
        hog_features = hog(image_gray, orientations=orientations, pixels_per_cell=pixels_per_cell)
        model_luoji = load('models/luojimodelHOG.pth')
        hog_features = hog_features.reshape(1, -1)
        feature_selector = SelectKBest(chi2, k=50)
        hog_features_new = feature_selector.fit_transform(hog_features, [0])  # 使用 [0] 作为目标值示例

        # Predict with each model
        prediction_luoji = model_luoji.predict(hog_features_new)
        cifar100_labels = unpickle("cifar-100-python/meta")['fine_label_names']
        c = Translator(from_lang="English", to_lang="Chinese").translate(cifar100_labels[prediction_luoji[0]])
        st.write("HOG特征的逻辑回归模型预测类别为：", prediction_luoji[0],c)
    elif optiontwo == "SIFT" and optionthree == "朴素贝叶斯模型":
        with open("temp_image.jpg", "wb") as file:
            file.write(uploaded_file.getvalue())
            # 加载保存的文件
        image = cv2.imread("temp_image.jpg")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        # 使用 PCA 进行特征降维到 50 维
        pca = PCA(n_components=50)
        reduced_features = pca.fit_transform(descriptors)
        # 加载模型
        model1 = load('models/beiyesimodelsift.pth')
        predicted_class1 = model1.predict(reduced_features)
        cifar100_labels = unpickle("cifar-100-python/meta")['fine_label_names']
        predicted_class1 = cifar100_labels[predicted_class1[0]]
        a = Translator(from_lang="English", to_lang="Chinese").translate(predicted_class1)
        st.write("SIFT特征朴素贝叶斯模型预测类别为：", predicted_class1, a)
        os.remove("temp_image.jpg")
    elif optiontwo == "SIFT" and optionthree == "KNN":
        with open("temp_image.jpg", "wb") as file:
            file.write(uploaded_file.getvalue())
            # 加载保存的文件
        image = cv2.imread("temp_image.jpg")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        # 使用 PCA 进行特征降维到 50 维
        pca = PCA(n_components=50)
        reduced_features = pca.fit_transform(descriptors)
        # 加载模型
        model2 = load('models/KNNmodelsift.pth')
        predicted_class2 = model2.predict(reduced_features)
        cifar100_labels = unpickle("cifar-100-python/meta")['fine_label_names']
        predicted_class2 = cifar100_labels[predicted_class2[0]]
        b = Translator(from_lang="English", to_lang="Chinese").translate(predicted_class2)
        st.write("SIFT特征KNN模型预测类别为：", predicted_class2, b)
        os.remove("temp_image.jpg")
    elif optiontwo == "SIFT" and optionthree == "逻辑回归":
        with open("temp_image.jpg", "wb") as file:
            file.write(uploaded_file.getvalue())
            # 加载保存的文件
        image = cv2.imread("temp_image.jpg")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        # 使用 PCA 进行特征降维到 50 维
        pca = PCA(n_components=50)
        reduced_features = pca.fit_transform(descriptors)
        # 加载模型
        model3 = load('models/luojimodelsift.pth')
        predicted_class3= model3.predict(reduced_features)
        cifar100_labels = unpickle("cifar-100-python/meta")['fine_label_names']
        predicted_class3 = cifar100_labels[predicted_class3[0]]
        c = Translator(from_lang="English", to_lang="Chinese").translate(predicted_class3)
        st.write("SIFT特征逻辑回归模型预测类别为：", predicted_class3, c)
        os.remove("temp_image.jpg")
st.header("准确率较低，仅供参考，欢迎指出问题")
st.header("联系方式QQ：3325949093")
