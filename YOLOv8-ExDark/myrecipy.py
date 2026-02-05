import os
from PIL import Image
import argparse
import shutil

# 标签对应的索引
labels = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']

def fix_image_profile(img):
    """转化成RGB格式，避免Libpng警告"""
    try:
        if img.mode != 'RGB':
            img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"Error fixing color profile: {e}")
        return None

def convert_to_jpg(img_path, output_path, quality=95):
    """同一数据集格式-jpg，可控制质量"""
    try:
        img = Image.open(img_path)
        img = fix_image_profile(img)
        if img is None:
            return None
            
        jpg_path = os.path.splitext(output_path)[0] + ".jpg"
        # 保存时优化质量和文件大小平衡
        img.save(jpg_path, quality=quality, optimize=True)
        
        # 验证转换后的文件
        if os.path.getsize(jpg_path) == 0:
            print(f"Warning: Empty file created for {img_path}")
            return None
            
        return jpg_path
    except Exception as e:
        print(f"Error converting {img_path} to JPG: {e}")
        return None

def check_and_log_conversion(original_path, converted_path):
    """检查并记录转换情况"""
    if os.path.exists(original_path) and os.path.exists(converted_path):
        orig_size = os.path.getsize(original_path) / (1024 * 1024)  # MB
        conv_size = os.path.getsize(converted_path) / (1024 * 1024)  # MB
        compression_ratio = conv_size / orig_size if orig_size > 0 else 0
        
        if compression_ratio < 0.3:  # 如果压缩率低于30%，可能质量损失过大
            print(f"Warning: High compression for {os.path.basename(original_path)}")
            print(f"  Original: {orig_size:.2f}MB -> Converted: {conv_size:.2f}MB (Ratio: {compression_ratio:.2%})")
        
        return compression_ratio
    return 0

def ExDark2Yolo(txts_dir: str, imgs_dir: str, ratio: str, version: int, output_dir: str, jpg_quality=95):
    """改进的数据集转换函数"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    ratios = ratio.split(':')
    ratio_train, ratio_test, ratio_val = int(ratios[0]), int(ratios[1]), int(ratios[2])
    ratio_sum = ratio_train + ratio_test + ratio_val
    dataset_perc = {'train': ratio_train / ratio_sum, 'test': ratio_test / ratio_sum, 'val': ratio_val / ratio_sum}

    # 创建子目录
    for t in dataset_perc:
        os.makedirs('/'.join([output_dir, t, 'images']), exist_ok=True)
        os.makedirs('/'.join([output_dir, t, 'labels']), exist_ok=True)

    total_original_size = 0
    total_converted_size = 0
    processed_count = 0
    skipped_count = 0

    for label in labels:
        print(f'Processing {label}...')
        label_txt_dir = '/'.join([txts_dir, label])
        
        if not os.path.exists(label_txt_dir):
            print(f"Warning: Label directory {label_txt_dir} does not exist")
            continue
            
        filenames = os.listdir(label_txt_dir)
        cur_idx = 0
        files_num = len(filenames)

        for filename in filenames:
            cur_idx += 1
            filename_no_ext = '.'.join(filename.split('.')[:-2])
            
            # 确定数据集划分
            if cur_idx < dataset_perc.get('train') * files_num:
                set_type = 'train'
            elif cur_idx < (dataset_perc.get('train') + dataset_perc.get('test')) * files_num:
                set_type = 'test'
            else:
                set_type = 'val'
                
            output_label_path = '/'.join([output_dir, set_type, 'labels', filename_no_ext + '.txt'])
            
            # 检查原始图像路径（支持多种格式）
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                potential_path = '/'.join([imgs_dir, label, filename_no_ext + ext])
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
            
            if img_path is None:
                print(f"Warning: Image file not found for {filename_no_ext}")
                skipped_count += 1
                continue

            # 转换图像格式
            jpg_path = convert_to_jpg(img_path, '/'.join([output_dir, set_type, 'images', filename_no_ext]), jpg_quality)
            if jpg_path is None:
                skipped_count += 1
                continue

            # 统计文件大小
            total_original_size += os.path.getsize(img_path)
            total_converted_size += os.path.getsize(jpg_path)
            processed_count += 1

            # 处理标注文件
            try:
                img = Image.open(jpg_path)
                width, height = img.size
                
                with open('/'.join([txts_dir, label, filename]), 'r') as txt:
                    with open(output_label_path, 'w') as yolo_output_file:
                        txt.readline()  # ignore first line
                        line = txt.readline()

                        while line != '':
                            datas = line.strip().split()
                            if len(datas) < 5:
                                line = txt.readline()
                                continue
                                
                            class_idx = labels.index(datas[0])
                            x0, y0, w0, h0 = int(datas[1]), int(datas[2]), int(datas[3]), int(datas[4])
                            
                            if version == 5:
                                x = (x0 + w0/2) / width
                                y = (y0 + h0/2) / height
                            elif version == 3:
                                x = x0 / width
                                y = y0 / height
                            else:
                                print("Version of YOLO error.")
                                return
                                
                            w = w0 / width
                            h = h0 / height

                            yolo_output_file.write(' '.join([str(class_idx),
                                                             format(x, '.6f'),
                                                             format(y, '.6f'),
                                                             format(w, '.6f'),
                                                             format(h, '.6f')]) + '\n')
                            line = txt.readline()

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                # 删除可能损坏的输出文件
                if os.path.exists(jpg_path):
                    os.remove(jpg_path)
                if os.path.exists(output_label_path):
                    os.remove(output_label_path)
                skipped_count += 1

    # 输出转换统计信息
    print(f"\n=== 转换统计 ===")
    print(f"处理图像数量: {processed_count}")
    print(f"跳过图像数量: {skipped_count}")
    print(f"原始总大小: {total_original_size/(1024 * 1024):.2f}MB")
    print(f"转换后总大小: {total_converted_size/(1024 * 1024):.2f}MB")
    print(f"压缩率: {total_converted_size/total_original_size*100:.1f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anndir', type=str, default='./ExDark/Annnotations', help="ExDark注释文件夹路径.")
    parser.add_argument('--imgdir', type=str, default='./ExDark/images', help="ExDark图像文件夹路径")
    parser.add_argument('--ratio', type=str, default='8:1:1', help="划分比率 train/test/val, default 8:1:1.")
    parser.add_argument('--version', type=int, choices=[3, 5], default=5, help="转化的YOLO版本")
    parser.add_argument('--output-dir', type=str, default="./datasets/ExDark", help="YOLO格式数据集输出的文件夹路径")
    parser.add_argument('--quality', type=int, default=95, help="JPEG质量 (1-100), 默认95")
    
    args = parser.parse_args()
    ExDark2Yolo(args.anndir, args.imgdir, args.ratio, args.version, args.output_dir, args.quality)
