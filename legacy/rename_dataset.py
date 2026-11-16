import os

def rename_dataset_images(base_dir='dataset_cropped'):
    """
    将指定目录下的所有图片重命名为 类别名_序号 的格式，避免中文路径问题。
    """
    categories = ['smiling', 'not_smiling']
    
    if not os.path.exists(base_dir):
        print(f"错误: 找不到目录 '{base_dir}'")
        return

    print(f"开始重命名目录 '{base_dir}' 中的图片...")

    for category in categories:
        category_path = os.path.join(base_dir, category)
        
        if not os.path.exists(category_path):
            print(f"跳过类别 '{category}': 找不到文件夹 {category_path}")
            continue

        # 获取文件夹下所有图片文件
        files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        files.sort() # 排序以保证重命名顺序一致

        count = 0
        for i, filename in enumerate(files):
            # 获取原文件后缀
            ext = os.path.splitext(filename)[1]
            # 构造新文件名，例如: smiling_0001.jpg
            new_name = f"{category}_{i+1:04d}{ext}"
            
            old_path = os.path.join(category_path, filename)
            new_path = os.path.join(category_path, new_name)

            try:
                # 执行重命名
                os.rename(old_path, new_path)
                count += 1
            except Exception as e:
                print(f"重命名文件 '{filename}' 失败: {e}")

        print(f"类别 '{category}' 重命名完成: 共处理 {count} 张图片")

if __name__ == "__main__":
    # 执行重命名逻辑
    rename_dataset_images('dataset_cropped')
    print("\n所有图片已成功转为英文数字格式！")
    print("现在您可以放心运行 train_smile_detector.py 进行训练了。")
