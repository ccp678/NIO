def split_text_file_by_lines(input_path, output_prefix, num_parts=100):
    """
    按行均匀拆分文本文件为100份
    :param input_path: 输入文件路径
    :param output_prefix: 输出文件前缀
    :param num_parts: 分割份数
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # 读取所有行（大文件慎用！）

    total_lines = len(lines)
    base_lines = total_lines // num_parts
    remainder = total_lines % num_parts

    start = 0
    for i in range(num_parts):
        end = start + base_lines + (1 if i < remainder else 0)
        part_lines = lines[start:end]

        output_path = f"{output_prefix}_{i + 1:03d}.txt"
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.writelines(part_lines)

        print(f"已生成: {output_path} (行数: {len(part_lines)})")
        start = end


# 示例：拆分 large_file.txt 为 100 份（按行）
split_text_file_by_lines("凡人修仙传.txt", "data/凡人修仙传_", 100)