import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#安全读取图像
def safe_read(file_path):
    clean_path = file_path.strip().strip('"').strip("'")

    if not os.path.exists(clean_path):
        print(f"提示：文件不存在，路径为 {clean_path}")
        return None

    try:
        image_data = np.fromfile(clean_path, dtype=np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return img
    except Exception as err:
        print(f"读取图片出错：{err}")
        return None

# 计算图像质心
def calculate_imagecentroid(img, img_width, img_height):
    _, binary_img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    img_moments = cv2.moments(binary_img)

    if img_moments["m00"] == 0:
        center_x = img_width // 2
        center_y = img_height // 2
        return center_x, center_y

    centroid_x = int(img_moments["m10"] / img_moments["m00"])
    centroid_y = int(img_moments["m01"] / img_moments["m00"])
    return centroid_x, centroid_y

# 脑图像配准与差异对比
def brain_imagealignment_compare(left_img_path, right_img_path, result_save_path):
    left_img = safe_read(left_img_path)
    right_img = safe_read(right_img_path)

    if left_img is None or right_img is None:
        print("图像读取失败，跳过本次处理")
        return

    img_height, img_width = left_img.shape[:2]
    right_imgresized = cv2.resize(right_img, (img_width, img_height))

    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_imgresized, cv2.COLOR_BGR2GRAY)

    right_gray_flipped = cv2.flip(right_gray, 1)

    left_cx, left_cy = calculate_imagecentroid(left_gray, img_width, img_height)
    right_cx, right_cy = calculate_imagecentroid(right_gray_flipped, img_width, img_height)

    affine_matrix = np.eye(2, 3, dtype=np.float32)
    affine_matrix[0, 2] = right_cx - left_cx
    affine_matrix[1, 2] = right_cy - left_cy

    print("开始执行图像精细配准...")
    try:
        stop_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-5)
        _, affine_matrix = cv2.findTransformECC(left_gray, right_gray_flipped, affine_matrix, cv2.MOTION_EUCLIDEAN, stop_criteria)
        print("精细配准完成")
    except cv2.error:
        print("精细配准未收敛，将使用粗配准结果")

    right_gray_aligned = cv2.warpAffine(right_gray_flipped, affine_matrix, (img_width, img_height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    left_blur = cv2.GaussianBlur(left_gray, (5, 5), 0)
    right_blur = cv2.GaussianBlur(right_gray_aligned, (5, 5), 0)

    diff_image = cv2.absdiff(left_blur, right_blur)
    max_diff_value = np.max(diff_image)  # 差异最大值（用于热力图归一化）

    high_diff_mask = (diff_image > 40).astype(np.uint8)
    black_bg_threshold = 5

    # 识别左右图像的黑背景区域
    left_black_bg = (left_gray <= black_bg_threshold)
    right_black_bg = (right_gray_aligned <= black_bg_threshold)
    total_black_bg = left_black_bg | right_black_bg  # 合并黑背景掩码

    # 生成处理前后的热力图
    heatmap_original = np.zeros_like(diff_image)
    heatmap_original[high_diff_mask == 1] = diff_image[high_diff_mask == 1]

    heatmap_processed = heatmap_original.copy()
    heatmap_processed[total_black_bg] = 0  # 去除黑背景区域的差异

    plot_figure = plt.figure(figsize=(18, 4.5), constrained_layout=True)
    grid_spec = plot_figure.add_gridspec(1, 4, wspace=0.02)

    # 原始左侧图像
    ax1 = plot_figure.add_subplot(grid_spec[0])
    ax1.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    ax1.set_title("1. 原始左侧", fontsize=11, pad=2)
    ax1.axis('off')
    ax1.set_xlim(0, img_width)
    ax1.set_ylim(img_height, 0)
    ax1.set_aspect('equal', adjustable='box')

    # 原始右侧图像
    ax2 = plot_figure.add_subplot(grid_spec[1])
    ax2.imshow(cv2.cvtColor(right_imgresized, cv2.COLOR_BGR2RGB))
    ax2.set_title("2. 原始右侧", fontsize=11, pad=2)
    ax2.axis('off')
    ax2.set_xlim(0, img_width)
    ax2.set_ylim(img_height, 0)
    ax2.set_aspect('equal', adjustable='box')

    # 处理前热力图
    ax3 = plot_figure.add_subplot(grid_spec[2])
    im3 = ax3.imshow(heatmap_original, cmap='jet', vmin=2, vmax=max_diff_value)
    ax3.set_title("3. 热力图（处理前）", fontsize=11, pad=2)
    ax3.axis('off')
    ax3.set_xlim(0, img_width)
    ax3.set_ylim(img_height, 0)
    ax3.set_aspect('equal', adjustable='box')

    # 处理后热力图
    ax4 = plot_figure.add_subplot(grid_spec[3])
    im4 = ax4.imshow(heatmap_processed, cmap='jet', vmin=2, vmax=max_diff_value)
    ax4.set_title("4. 热力图（处理后）", fontsize=11, pad=2)
    ax4.axis('off')
    ax4.set_xlim(0, img_width)
    ax4.set_ylim(img_height, 0)
    ax4.set_aspect('equal', adjustable='box')

    color_bar = plot_figure.colorbar(im4, ax=[ax3, ax4], fraction=0.02, pad=0.01, shrink=0.8)
    color_bar.set_label('Difference', fontsize=9)

    # 保存结果图像
    plt.savefig(result_save_path, dpi=150, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close()


# 批量处理脑图像
def batch_process_brain_images(left_folder, right_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹：{output_folder}")

    valid_image_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    left_image_files = [f for f in os.listdir(left_folder) if f.lower().endswith(valid_image_ext)]

    if len(left_image_files) == 0:
        print(f"在文件夹 {left_folder} 中未找到有效图片")
        return

    # 开始批量处理
    print(f"共找到 {len(left_image_files)} 张左侧图像，开始批量处理...")
    print("-" * 40)

    for left_file_name in left_image_files:
        right_file_name = left_file_name.replace('left', 'right')

        # 拼接完整路径
        left_full_path = os.path.join(left_folder, left_file_name)
        right_full_path = os.path.join(right_folder, right_file_name)

        if not os.path.exists(right_full_path):
            print(f"跳过 {left_file_name}：未找到对应的右侧图片")
            continue

        output_file_name = left_file_name.replace('left', 'SUV')
        output_full_path = os.path.join(output_folder, output_file_name)

        print(f"正在处理：{left_file_name} <-> {right_file_name}")
        brain_imagealignment_compare(left_full_path, right_full_path, output_full_path)
        print(f"处理完成，结果已保存至：{output_file_name}")
        print("-" * 40)

    print("所有图像批量处理完成！")


if __name__ == "__main__":

    input_left_dir = r"C:\Users\Geralt\Desktop\大创\PythonProject11\3.16\upper_left"
    input_right_dir = r"C:\Users\Geralt\Desktop\大创\PythonProject11\3.16\upper_right"
    output_result_dir = r"C:\Users\Geralt\Desktop\大创\PythonProject11\3.16\SUV-up"

    batch_process_brain_images(input_left_dir, input_right_dir, output_result_dir)
