import re
import pandas as pd
from collections import defaultdict

# 读取武器列表文件
def read_weapon_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    weapons = re.findall(r'\d+\.\s*\*\*(.*?)\*\*：(.*?)\n', content, re.DOTALL)
    return pd.DataFrame(weapons, columns=['name', 'description'])

# 文本预处理函数
def preprocess_text(text):
    # 移除特殊符号和多余空格
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# 基于规则的武器提取
def extract_weapons_by_rules(text):
    weapon_patterns = [
        r'([\u4e00-\u9fa5]+[剑刀枪棍符箓幡镜瓶珠鼎印])(?:\s*[，,。；;])',
        r'(?:使用|祭出|持有|获得|得到)\s*([\u4e00-\u9fa5]+[法器法宝符箓])'
    ]
    weapons = set()
    for pattern in weapon_patterns:
        matches = re.findall(pattern, text)
        weapons.update(matches)
    return list(weapons)

# 生成BIO格式训练数据
def generate_bio_data(text, known_weapons):
    tokens = list(text)
    labels = ['O'] * len(tokens)
    # 标记已知武器
    for weapon in known_weapons:
        start = text.find(weapon)
        if start != -1:
            labels[start] = 'B-WEAPON'
            for i in range(start+1, start+len(weapon)):
                labels[i] = 'I-WEAPON'
    return list(zip(tokens, labels))

# 主函数
def main():
    # 读取武器列表
    weapon_df = read_weapon_list('韩立武器列表.txt')
    known_weapons = set(weapon_df['name'])
    
    # 模拟读取全文本（实际应用需读取完整文件）
    with open('凡人修仙传.txt', 'r', encoding='utf-8') as f:
        full_text = f.read(100000)  # 读取部分文本示例
    
    # 规则提取新武器
    new_weapons = extract_weapons_by_rules(full_text)
    all_weapons = known_weapons.union(set(new_weapons))

    
    # 生成训练数据
    bio_data = generate_bio_data(full_text, all_weapons)
    
    # 保存结果
    pd.DataFrame(bio_data, columns=['token', 'label']).to_csv('weapon_ner_train.csv', index=False)
    pd.DataFrame(list(all_weapons), columns=['weapon']).to_csv('all_extracted_weapons.csv', index=False)
    
    print(f"提取武器总数: {len(all_weapons)}")
    print(all_weapons)
    print(f"生成NER训练数据行数: {len(bio_data)}")

if __name__ == "__main__":
    main()