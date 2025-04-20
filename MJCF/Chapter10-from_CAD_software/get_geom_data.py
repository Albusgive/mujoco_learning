import os
import csv
import re
from pypinyin import lazy_pinyin, Style

def chinese_to_pinyin(name):
    # 将文件名转换为拼音格式（保持不变）
    parts = re.split(r'([\u4e00-\u9fff]+)', name)
    pinyin_parts = []
    for part in parts:
        if not part:
            continue
        if re.search(r'[\u4e00-\u9fff]', part):
            pinyin = '_'.join(lazy_pinyin(part, style=Style.NORMAL))
            pinyin_parts.append(pinyin)
        else:
            cleaned = re.sub(r'[^a-zA-Z0-9]', '_', part).lower()
            pinyin_parts.append(cleaned)
    return '_'.join(filter(None, pinyin_parts))

def process_files(csv_file, stl_ext, target_dir='.'):
    # 修改1: 使用 GBK 编码读取 CSV 文件（适配中文环境生成的 CSV）
    mass_data = {}
    with open(csv_file, 'r', encoding='gbk') as f:  # 重点修改这里！！！
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2: 
                continue
            raw_name = row[0].strip()
            pinyin_name = chinese_to_pinyin(raw_name)
            try:
                mass = float(row[1]) / 1000  # 转换为千克
            except ValueError:
                mass = 0
            mass_data[pinyin_name] = mass

    # 后续代码保持不变...
    results = []
    for filename in os.listdir(target_dir):
        filepath = os.path.join(target_dir, filename)
        
        if not os.path.isfile(filepath):
            continue
        if not filename.lower().endswith(stl_ext):
            continue

        base_name = os.path.splitext(filename)[0]
        new_base = chinese_to_pinyin(base_name)
        new_filename = f"{new_base}.stl"
        matched_mass = next((v for k, v in mass_data.items() if k in new_base), 0.0001)
        results.append({
            "original_name": filepath, 
            "pinyin_name": new_base,
            "new_filename": new_filename,
            "mass_kg": f"{matched_mass:.4f}"
        })

    return results

# 其余代码保持不变...
CSV_FILE = 'mass.csv'
STL_EXT = ('.stl', '.STL')
target_dir = [
        "base", 
        "left_hip", 
        "left_thigh", 
        "left_calf", 
        "right_hip", 
        "right_thigh", 
        "right_calf", 
    ]

for dir in target_dir:
    output_data = process_files(CSV_FILE, STL_EXT, dir)
    os.makedirs('.', exist_ok=True)
    geom_data_filename = 'geom_data_'+dir+'.csv'
    with open(os.path.join('.',geom_data_filename), 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=["original_name", "pinyin_name", "new_filename", "mass_kg"])
        writer.writeheader()
        writer.writerows(output_data)
    print(f"处理完成！结果已保存至：{os.path.abspath(geom_data_filename)}")