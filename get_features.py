import os
import cv2
import numpy as np
import onnxruntime as ort

def preprocess_image(image_path, input_size=(112, 112)):
    """
    加载并预处理图片为模型输入格式
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_size)
    image = image.astype('float32')
    image = (image / 255.0 - 0.5) / 0.5  # 标准化到 [-1, 1]
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    return image

def extract_features(model, image_batch):
    """
    使用 ONNX 模型提取特征
    """
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    return model.run([output_name], {input_name: image_batch})[0]

def process_images_and_extract_features(image_folder, model_path, batch_size, output_folder):
    """
    从原始图片中提取特征并保存
    """
    print(f"开始特征提取，图片路径：{image_folder}，模型路径：{model_path}，输出路径：{output_folder}")

    # 加载 ONNX 模型
    model = ort.InferenceSession(model_path)

    # 获取所有图片路径
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]
    print(f"共找到 {len(image_paths)} 张图片")
    os.makedirs(output_folder, exist_ok=True)

    batch = []
    file_names = []

    for idx, image_path in enumerate(image_paths, 1):
        try:
            # 预处理图片
            image = preprocess_image(image_path)
            batch.append(image)
            file_names.append(os.path.basename(image_path))

            # 如果达到批处理大小，则进行特征提取
            if len(batch) == batch_size:
                batch_array = np.stack(batch)  # 转为 (N, C, H, W)
                features = extract_features(model, batch_array)

                # 保存特征
                for i, feature in enumerate(features):
                    output_path = os.path.join(output_folder, file_names[i].replace(".jpg", ".npy").replace(".png", ".npy"))
                    np.save(output_path, feature)

                print(f"已处理 {idx} / {len(image_paths)} 张图片")
                batch.clear()
                file_names.clear()

        except ValueError as e:
            print(e)

    # 处理剩余图片
    if batch:
        batch_array = np.stack(batch)
        features = extract_features(model, batch_array)
        for i, feature in enumerate(features):
            output_path = os.path.join(output_folder, file_names[i].replace(".jpg", ".npy").replace(".png", ".npy"))
            np.save(output_path, feature)
        print(f"已处理剩余 {len(batch)} 张图片")

    print("特征提取完成！")

if __name__ == "__main__":
    # 参数配置
    folder_path = './processed'  # 已预处理的图片路径
    model_path = './antelopev2/glintr100.onnx'  # ONNX 模型路径
    output_folder = './features'  # 保存特征的路径
    batch_size = 32  # 批处理大小

    process_images_and_extract_features(folder_path, model_path, batch_size, output_folder)