import os
import cv2
import dlib
import numpy as np
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def detect_and_align_faces(image_path, predictor_path, target_size=(112, 112)):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图片: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    if len(rects) == 0:
        print(f"未检测到人脸: {image_path}")
        return []

    aligned_faces = []
    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = np.array([[p.x, p.y] for p in shape.parts()])
        
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        face = image[y1:y2, x1:x2]
        aligned_face = cv2.resize(face, target_size)
        aligned_faces.append(aligned_face)

    return aligned_faces

def process_single_image(image_path, output_folder, predictor_path):
    image_name = os.path.basename(image_path)
    
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

def process_folder_for_preprocessing(input_folder, output_folder, predictor_path, max_workers=2, batch_size=5, start_index=31551):
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有图片路径并排序
    image_paths = sorted([os.path.join(input_folder, img) for img in os.listdir(input_folder) if img.endswith(('.jpg', '.png'))])

    # 从 start_index 开始遍历
    image_paths = image_paths[start_index:]

    with tqdm(total=len(image_paths), desc="处理图片", unit="张") as pbar:
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_single_image, img_path, output_folder, predictor_path): img_path for img_path in batch
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"图片处理任务失败: {e}")
            
            pbar.update(len(batch))
            gc.collect()

if __name__ == "__main__":
    input_folder = "./Gallery"  # 输入图片文件夹路径
    output_folder = "./processed"  # 输出裁剪后人脸文件夹路径
    predictor_path = "./shape_predictor_68_face_landmarks.dat"  # Dlib 预测器文件路径

    max_workers = 2  # 适用于 2 核 CPU 的设置

    # 批量处理图片，从第31552个文件开始
    process_folder_for_preprocessing(input_folder, output_folder, predictor_path, max_workers=max_workers, start_index=1)
