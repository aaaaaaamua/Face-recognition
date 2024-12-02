import os
import cv2
import dlib
import numpy as np
import onnxruntime as ort
from scipy.spatial.distance import cosine  # 用于计算余弦相似度
from tqdm import tqdm  # 用于显示进度条

# 人脸检测和预处理
def preprocess_and_align_face(image_path, predictor_path, target_size=(112, 112)):
    """
    加载图片并进行预处理，检测人脸，裁剪并对齐人脸
    :param image_path: 图片路径
    :param predictor_path: Dlib 人脸关键点预测器路径
    :param target_size: 裁剪后的人脸目标尺寸
    :return: 预处理后的人脸图像
    """
    print(f"正在处理: {image_path}")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # 加载图片并转换为灰度图
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图片: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    rects = detector(gray, 1)
    if len(rects) == 0:
        print(f"未检测到人脸: {image_path}")
        return None

    # 获取第一个人脸（可以根据需求调整处理多个面部）
    rect = rects[0]  # 假设只有一个人脸
    shape = predictor(gray, rect)

    # 获取人脸框
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

    # 添加缓冲区，扩大裁剪框
    padding = 20
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    # 裁剪人脸并调整尺寸
    face = image[y1:y2, x1:x2]
    aligned_face = cv2.resize(face, target_size)
    
    print(f"完成处理: {image_path}")
    return aligned_face

# 提取特征向量
def extract_feature(model, image):
    """
    使用 ONNX 模型提取特征向量
    :param model: ONNX 模型
    :param image: 输入图像（已预处理并调整为目标尺寸）
    :return: 提取的人脸特征向量
    """
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    # 预处理图片（标准化到 [-1, 1]，并转换为 CHW 格式）
    image = image.astype('float32')
    image = (image / 255.0 - 0.5) / 0.5  # 标准化到 [-1, 1]
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)  # 添加 batch 维度

    # 提取特征向量
    feature = model.run([output_name], {input_name: image})[0]
    return feature.flatten()  # 返回一维特征向量

# 加载特征向量库
def load_feature_vectors(feature_folder):
    """
    加载存储的特征向量
    :param feature_folder: 存储特征向量的文件夹路径
    :return: 特征向量字典，键是文件名，值是对应的特征向量
    """
    feature_vectors = {}
    print(f"加载特征库中的特征向量...")
    for file_name in os.listdir(feature_folder):
        if file_name.endswith('.npy'):
            feature_path = os.path.join(feature_folder, file_name)
            feature_vector = np.load(feature_path)
            feature_vectors[file_name] = feature_vector
    print(f"特征库加载完成，包含 {len(feature_vectors)} 个特征向量")
    return feature_vectors

# 计算余弦相似度并进行人脸匹配
def match_face(uploaded_image_path, feature_folder, model, top_n=5):
    """
    对上传图片进行人脸识别，计算与数据库中人脸的相似度，并返回最相似的前N个
    :param uploaded_image_path: 上传的图片路径
    :param feature_folder: 存储特征向量的文件夹路径
    :param model: 预训练的 ONNX 模型
    :param top_n: 返回前N个最相似的匹配结果
    :return: 最相似的前N个图片文件名和相似度
    """
    # 对上传的图片进行预处理并提取特征向量
    print(f"提取上传图片特征: {uploaded_image_path}")
    aligned_face = preprocess_and_align_face(uploaded_image_path, predictor_path)
    if aligned_face is None:
        return None, []

    uploaded_feature = extract_feature(model, aligned_face)

    # 加载数据库中的特征向量
    feature_vectors = load_feature_vectors(feature_folder)

    # 计算与数据库中每个特征向量的相似度
    similarity_list = []
    print("开始匹配相似的人脸...")
    for file_name, stored_feature in tqdm(feature_vectors.items(), desc="匹配人脸", unit="个"):
        similarity = 1 - cosine(uploaded_feature, stored_feature)  # 计算余弦相似度
        similarity_list.append((file_name, similarity))

    # 按相似度排序并取前N个
    similarity_list.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarity_list[:top_n]

    # 输出前N个匹配结果
    print(f"最相似的前{top_n}个匹配结果:")
    for idx, (file_name, similarity) in enumerate(top_matches):
        print(f"第{idx+1}个匹配: {file_name}，相似度: {similarity:.4f}")

    return top_matches

if __name__ == "__main__":
    # 配置参数
    uploaded_image_path = './uploaded_image.jpg'  # 上传图片路径
    feature_folder = './features'  # 存储特征向量的路径
    model_path = './antelopev2/glintr100.onnx'  # ONNX 模型路径
    predictor_path = "./shape_predictor_68_face_landmarks.dat"  # Dlib 预测器路径
    top_n = 5  # 返回最相似的前5个匹配结果

    # 加载 ONNX 模型
    model = ort.InferenceSession(model_path)

    # 进行人脸识别并匹配
    match_result = match_face(uploaded_image_path, feature_folder, model, top_n=top_n)

    if match_result:
        print("匹配完成！")
    else:
        print("未找到匹配的人脸")
