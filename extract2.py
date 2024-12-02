import os
import cv2
import dlib
import numpy as np
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# 用于人脸检测和裁剪的函数
def detect_and_align_faces(image_path, predictor_path, target_size=(112, 112)):
    """
    检测人脸并裁剪，返回对齐后的人脸图像列表。
    :param image_path: 输入图片路径
    :param predictor_path: 关键点预测器路径
    :param target_size: 裁剪后人脸的目标尺寸 (H, W)
    :return: 裁剪并对齐后的人脸图像列表
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # 加载图片并转灰度图
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图片: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    rects = detector(gray, 1)
    if len(rects) == 0:
        print(f"未检测到人脸: {image_path}")
        return []

    aligned_faces = []
    for rect in rects:
        # 获取关键点
        shape = predictor(gray, rect)
        shape_np = np.array([[p.x, p.y] for p in shape.parts()])

        # 获取原始框坐标
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

        # 添加缓冲区，扩大裁剪框
        padding = 20  # 你可以调整这个值，增加/减少裁剪的区域
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        # 裁剪人脸
        face = image[y1:y2, x1:x2]

        # 调整到目标尺寸
        aligned_face = cv2.resize(face, target_size)
        aligned_faces.append(aligned_face)

    return aligned_faces

# 用于处理单张图片并保存裁剪结果
def process_single_image(image_path, output_folder, predictor_path):
    """
    处理单张图片，检测并裁剪人脸，保存结果。
    :param image_path: 输入图片路径
    :param output_folder: 输出裁剪后图片文件夹
    :param predictor_path: Dlib 人脸关键点预测器路径
    """
    image_name = os.path.basename(image_path)
    
    # 检查是否已经处理过该图片
    output_files = [f for f in os.listdir(output_folder) if f.startswith(os.path.splitext(image_name)[0])]
    if output_files:
        print(f"图片 {image_name} 已处理过，跳过该图片。")
        return
    
    try:
        faces = detect_and_align_faces(image_path, predictor_path)
        for i, face in enumerate(faces):
            output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_face_{i}.jpg")
            cv2.imwrite(output_path, face)
            print(f"已保存裁剪人脸: {output_path}")
    except Exception as e:
        print(f"处理图片 {image_name} 时出错: {e}")

# 用于批量处理图片（多线程版）
def process_folder_for_preprocessing(input_folder, output_folder, predictor_path, max_workers=2, batch_size=5):
    """
    使用多线程批量检测人脸并保存对齐后的结果（仅预处理）。
    :param input_folder: 输入图片文件夹
    :param output_folder: 输出裁剪后图片文件夹
    :param predictor_path: Dlib 人脸关键点预测器路径
    :param max_workers: 最大线程数
    :param batch_size: 每批处理的图片数量
    """
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有图片路径
    image_paths = [os.path.join(input_folder, img) for img in os.listdir(input_folder) if img.endswith(('.jpg', '.png'))]

    # 批量处理图片
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_image, img_path, output_folder, predictor_path): img_path for img_path in batch
            }
            for future in as_completed(futures):
                try:
                    future.result()  # 获取任务结果
                except Exception as e:
                    print(f"图片处理任务失败: {e}")

        # 强制进行垃圾回收，释放内存
        gc.collect()

if __name__ == "__main__":
    # 配置路径
    input_folder = "./Gallery"  # 输入图片文件夹路径
    output_folder = "./processed"  # 输出裁剪后人脸文件夹路径
    predictor_path = "./shape_predictor_68_face_landmarks.dat"  # Dlib 预测器文件路径

    # 最大线程数（根据 CPU 性能调整）
    max_workers = 2  # 适用于 2 核 CPU 的设置

    # 批量处理图片
    process_folder_for_preprocessing(input_folder, output_folder, predictor_path, max_workers=max_workers)
