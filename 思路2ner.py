import re
import json
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel



# 武器实体识别模型
class WeaponNERModel(nn.Module):
    def __init__(self, bert_model_name='hfl/chinese-bert-wwm-ext', num_tags=5):
        super(WeaponNERModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(512, num_tags)  # BIOES标签

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state

        # BiLSTM处理
        lstm_output, _ = self.bilstm(sequence_output)

        # CRF输出
        logits = self.classifier(self.dropout(lstm_output))
        return logits


# 关系抽取模型
class RelationExtractionModel(nn.Module):
    def __init__(self, bert_model_name='hfl/chinese-bert-wwm-ext', num_relations=5):
        super(RelationExtractionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=self.bert.config.hidden_size,
            num_heads=8,
            batch_first=True
        )

        # 关系分类器
        self.classifier = nn.Linear(self.bert.config.hidden_size * 3, num_relations)

    def forward(self, input_ids, attention_mask, token_type_ids, entity_positions):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state

        # 应用注意力机制
        attn_output, _ = self.attention(sequence_output, sequence_output, sequence_output)

        # 提取实体表示
        batch_relations = []
        for i, positions in enumerate(entity_positions):
            if len(positions) >= 2:
                # 获取主角和武器的表示
                char_start, char_end = positions[0]
                weapon_start, weapon_end = positions[1]

                char_rep = attn_output[i, char_start:char_end + 1].mean(dim=0)
                weapon_rep = attn_output[i, weapon_start:weapon_end + 1].mean(dim=0)

                # 计算上下文表示
                context_rep = attn_output[i].mean(dim=0)

                # 组合特征
                combined = torch.cat([char_rep, weapon_rep, context_rep], dim=-1)
                relation_logits = self.classifier(self.dropout(combined))
                batch_relations.append(relation_logits)

        return torch.stack(batch_relations) if batch_relations else torch.tensor([])


class WeaponExtractor:

    def __init__(self, model_path=None):
        # model_name = 'hfl/chinese-bert-wwm-ext'
        # model_name = "model/chinese-bert-wwm-ext"
        model_name = 'model/bert-base-chinese-finetuned-ner'
        # model_name = "ckiplab/bert-base-chinese-ner"
        # model_path = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # 初始化模型
        self.ner_model = WeaponNERModel()
        self.relation_model = RelationExtractionModel()

        if model_path:
            self.load_model(model_path)

        self.ner_model.to(self.device)
        self.relation_model.to(self.device)


    def load_model(self, model_path):
        """加载预训练模型"""
        state_dict = torch.load(model_path, map_location=self.device)
        self.ner_model.load_state_dict(state_dict['ner_model'])
        self.relation_model.load_state_dict(state_dict['relation_model'])


    def preprocess_text(self, text):
        """文本预处理"""
        # 清洗文本
        text = re.sub(r'[^\u4e00-\u9fff\s，。！？；：、（）《》【】「」]', '', text)

        # 分句
        sentences = re.split(r'[。！？；：]', text)
        return [s.strip() for s in sentences if len(s.strip()) > 3]


    def extract_weapons_advanced(self, text):
        """针对修仙小说的武器提取"""
        print("开始提取")
        sentences = self.preprocess_text(text)
        results = []

        for sentence in sentences:
            # 使用NER模型识别实体
            ner_result = self.predict_ner(sentence)

            # 查找武器和角色
            weapons = []
            characters = []

            for entity in ner_result:
                if entity['type'] == 'WEAPON':
                    weapons.append((entity['start'], entity['end'], entity['text']))
                elif entity['type'] == 'PERSON':
                    characters.append((entity['start'], entity['end'], entity['text']))

            # 如果找到武器和角色，进行关系抽取
            if weapons and characters:
                # 选择主要角色（韩立）
                main_char = None
                for char in characters:
                    if '韩立' in char[2] or '厉飞雨' in char[2]:  # 主要角色
                        main_char = char
                        break

                if not main_char and characters:
                    main_char = characters[0]  # 默认第一个角色

                # 对每个武器检查与主角的关系
                for weapon in weapons:
                    relation = self.predict_relation(
                        sentence,
                        (main_char[0], main_char[1]),
                        (weapon[0], weapon[1])
                    )

                    if relation == 'USE':  # 使用关系
                        results.append({
                            'sentence': sentence,
                            'character': main_char[2],
                            'weapon': weapon[2],
                            'weapon_type': self.weapon_dict.weapon_to_type.get(weapon[2], '未知'),
                            'confidence': 0.95  # 置信度
                        })

        return results


    def predict_ner(self, sentence):
        """预测命名实体"""
        # 简化的NER预测实现
        inputs = self.tokenizer(sentence, max_length=512, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.ner_model(**inputs)
            predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

        # 将预测转换为实体
        entities = []
        current_entity = None

        for i, tag_id in enumerate(predictions):
            if tag_id == 1:  # B-WEAPON
                if current_entity:
                    entities.append(current_entity)
                current_entity = {'start': i, 'end': i, 'type': 'WEAPON', 'text': ''}
            elif tag_id == 2 and current_entity:  # I-WEAPON
                current_entity['end'] = i
            elif tag_id == 0 and current_entity:  # O
                entities.append(current_entity)
                current_entity = None

        # 提取实体文本
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        for entity in entities:
            entity_tokens = tokens[entity['start']:entity['end'] + 1]
            entity['text'] = self.tokenizer.convert_tokens_to_string(entity_tokens).replace(' ', '')

        return entities


    def predict_relation(self, sentence, char_position, weapon_position):
        """预测关系"""
        # 简化的关系预测实现
        inputs = self.tokenizer(sentence,max_length=512 ,return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.relation_model(
                inputs['input_ids'],
                inputs['attention_mask'],
                inputs['token_type_ids'],
                [char_position, weapon_position]
            )

            if len(logits) > 0:
                prediction = torch.argmax(logits, dim=-1)
                return ['NO_RELATION', 'USE', 'OWN', 'CREATE', 'LOSE'][prediction.item()]

        return 'NO_RELATION'


# 使用示例
if __name__ == "__main__":
    # 示例文本（来自《凡人修仙传》）

    # with open("凡人修仙传.txt", "r", encoding="utf-8") as f:
    #     text = f.read()
    text = """
    韩立祭出青竹蜂云剑，七十二口飞剑化作漫天青光向敌人斩去。同时他催动风雷翅，
    身形如电般在空中闪烁。面对强敌，韩立又取出玄天斩灵剑，这把玄天之宝威力无穷。
    厉飞雨则施展千刃术，无数刀光剑影笼罩战场。
    """

    # 创建抽取器
    extractor = WeaponExtractor()

    # 提取武器信息
    results = extractor.extract_weapons_advanced(text)
    with open("weapon_results.txt", "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"{result['character']} -> {result['weapon']} ({result['weapon_type']})\n")
            f.write(f"上下文: {result['sentence']}\n")
            f.write("\n")

    # 打印结果
    print("提取到的武器使用关系:")
    for result in results:
        print(f"{result['character']} -> {result['weapon']} ({result['weapon_type']})")
        print(f"上下文: {result['sentence']}")
        print()