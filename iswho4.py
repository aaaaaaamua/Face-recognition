import os
import cv2
import dlib
import numpy as np
import onnxruntime as ort
from scipy.spatial.distance import cosine
from tqdm import tqdm


def preprocess_and_align_face(image_path, predictor_path, target_size=(112, 112)):
    """
    加载图片并进行预处理，检测人脸，按顺序修复旋转和畸变，并对齐
    """
    print(f"正在处理: {image_path}")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图片: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) == 0:
        print(f"未检测到人脸: {image_path}")
        return None

    rect = rects[0]
    shape = predictor(gray, rect)
    points = np.array([[p.x, p.y] for p in shape.parts()])

    # 修复旋转
    if is_face_sideways(points):
        print(f"检测到人脸不正，进行旋转矫正: {image_path}")
        image, points = correct_sideways_face(image, points)

    # 修复畸变
    if is_face_distorted(points, image.shape):
        print(f"检测到人脸畸变，尝试修复: {image_path}")
        try:
            corrected_image = correct_face_perspective(image, points)
            gray_corrected = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
            rects_corrected = detector(gray_corrected, 1)
            if len(rects_corrected) > 0:
                rect = rects_corrected[0]
                shape = predictor(gray_corrected, rect)
                points = np.array([[p.x, p.y] for p in shape.parts()])
                image = corrected_image
                print(f"修复成功: {image_path}")
        except Exception as e:
            print(f"修复失败，使用原图继续: {e}")

    # 对齐裁剪
    aligned_face = align_face(image, points, target_size)
    return aligned_face


def is_face_sideways(points):
    left_eye = np.mean(points[36:42], axis=0)
    right_eye = np.mean(points[42:48], axis=0)
    eye_slope = abs((right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0] + 1e-6))
    return eye_slope > 0.5


def correct_sideways_face(image, points):
    left_eye = np.mean(points[36:42], axis=0)
    right_eye = np.mean(points[42:48], axis=0)
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    eye_center = ((left_eye + right_eye) / 2).astype("int")
    rot_matrix = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale=1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))

    ones = np.ones((points.shape[0], 1))
    points_augmented = np.hstack([points, ones])
    new_points = np.dot(rot_matrix, points_augmented.T).T
    return rotated_image, new_points


def is_face_distorted(points, image_shape):
    img_h, img_w = image_shape[:2]
    left_eye = np.mean(points[36:42], axis=0)
    right_eye = np.mean(points[42:48], axis=0)
    eye_distance = np.linalg.norm(left_eye - right_eye)
    face_width = np.max(points[:, 0]) - np.min(points[:, 0])
    return eye_distance / face_width < 0.2 or np.min(points[:, 0]) < 5 or np.max(points[:, 0]) > img_w - 5


def correct_face_perspective(image, points):
    left_eye = np.mean(points[36:42], axis=0).astype("int")
    right_eye = np.mean(points[42:48], axis=0).astype("int")
    nose_tip = points[33].astype("int")
    mouth_center = np.mean(points[48:68], axis=0).astype("int")

    src = np.array([left_eye, right_eye, nose_tip, mouth_center], dtype="float32")
    dst = np.array([[50, 50], [150, 50], [100, 150], [100, 200]], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    corrected_image = cv2.warpPerspective(image, M, (200, 250))
    return corrected_image


def align_face(image, points, target_size):
    left_eye = np.mean(points[36:42], axis=0)
    right_eye = np.mean(points[42:48], axis=0)
    eye_center = (left_eye + right_eye) // 2
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    rot_matrix = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale=1.0)
    rotated = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    x, y, w, h = cv2.boundingRect(points)
    cropped = rotated[y:y + h, x:x + w]
    aligned_face = cv2.resize(cropped, target_size)
    return aligned_face


def extract_feature(model, image):
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    image = image.astype('float32')
    image = (image / 255.0 - 0.5) / 0.5
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    feature = model.run([output_name], {input_name: image})[0]
    return feature.flatten()


def load_feature_vectors(feature_folder):
    feature_vectors = {}
    for file_name in os.listdir(feature_folder):
        if file_name.endswith('.npy'):
            feature_path = os.path.join(feature_folder, file_name)
            feature_vectors[file_name] = np.load(feature_path)
    return feature_vectors


def match_face(uploaded_image_path, feature_folder, model, predictor_path, top_n=5):
    aligned_face = preprocess_and_align_face(uploaded_image_path, predictor_path)
    if aligned_face is None:
        print("未检测到有效人脸")
        return

    uploaded_feature = extract_feature(model, aligned_face)
    feature_vectors = load_feature_vectors(feature_folder)

    similarity_list = []
    for file_name, stored_feature in tqdm(feature_vectors.items(), desc="匹配人脸", unit="个"):
        similarity = 1 - cosine(uploaded_feature, stored_feature)
        similarity_list.append((file_name, similarity))

    similarity_list.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarity_list[:top_n]

    print(f"最相似的前{top_n}个匹配结果:")
    for idx, (file_name, similarity) in enumerate(top_matches):
        print(f"第{idx + 1}个匹配: {file_name}，相似度: {similarity:.4f}")


if __name__ == "__main__":
    uploaded_image_path = './uploaded_image.jpg'
    feature_folder = './features'
    model_path = './antelopev2/glintr100.onnx'
    predictor_path = "./shape_predictor_68_face_landmarks.dat"
    top_n = 5

    model = ort.InferenceSession(model_path)
    match_face(uploaded_image_path, feature_folder, model, predictor_path, top_n=top_n)