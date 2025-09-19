import requests
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import os

# 确保你已经设置了环境变量 DEEPSEEK_API_KEY，或者在代码中直接替换你的API密钥
# 可以从 https://platform.deepseek.com/ 获取API Key
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-aa47d6c050eb42b8aba94323e362bec9")
DEEPSEEK_API_KEY = "sk-aa47d6c050eb42b8aba94323e362bec9"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"  # 请确认DeepSeek API的最新端点

# 1. 定义数据模型：我们希望从文本中提取的武器信息
class Weapon(BaseModel):
    """武器信息提取模型"""
    name: str = Field(description="武器的正式名称，例如 '青竹蜂云剑', '金蚨子母刃'")
    type: str = Field(description="武器的类型或类别，例如 '本命法宝', '飞行法宝', '顶阶攻击法器', '符宝'")
    description: Optional[str] = Field(default=None, description="武器的功能、特点、威力或简要描述。若文本未详细描述则返回null")
    source: Optional[str] = Field(default=None, description="武器的来源，例如 '炼制', '夺宝', '赠送', '兑换'。若文本未提及则返回null")
    status: Optional[str] = Field(default=None, description="武器的状态，例如 '现存', '已损毁', '已升级'。若文本未提及则返回null")

class WeaponExtractionResult(BaseModel):
    """武器信息提取容器"""
    weapons: List[Weapon] = Field(description="从文本中提取到的武器信息列表。若无任何武器信息，则返回空列表")

# 2. 构建系统提示词 (System Prompt)
SYSTEM_PROMPT = """你是一个专业的信息提取算法，专门从网络小说文本中提取武器装备的结构化信息。
请严格仅从用户提供的文本中提取相关信息。
如果文本中没有明确提及被要求提取的属性的值，请将该属性的值设为 null。
你必须将输出格式化为一个严格的JSON对象，该JSON对象必须能够被解析并匹配以下Pydantic模型结构：
{
  "weapons": [
    {
      "name": "武器名称",
      "type": "武器类型",
      "description": "武器描述或null",
      "source": "武器来源或null",
    }
  ]
}
确保JSON是有效的，并且只包含模型指定的字段。
"""

# 3. 示例文本和期望输出 (Few-shot Learning)，用于帮助模型更好地理解任务
EXAMPLES = [
    {
        "role": "user",
        "content": "韩立从陆师兄那里缴获了青蛟旗，这是一件顶阶攻击法器，灵气可化为蛟龙进行攻击。"
    },
    {
        "role": "assistant",
        "content": '{\n  "weapons": [\n    {\n      "name": "青蛟旗",\n      "type": "顶阶攻击法器",\n      "description": "灵气可化为蛟龙进行攻击",\n      "source": "缴获",\n   }\n  ]\n}'
    },
    {
        "role": "user",
        "content": "韩立用千年草药在万宝楼换来了金蚨子母刃，这是一母八子的成套飞刃，锋利无比，但后来在与强敌斗法中损毁了。"
    },
    {
        "role": "assistant",
        "content": '{\n  "weapons": [\n    {\n      "name": "金蚨子母刃",\n      "type": "顶阶攻击法器",\n      "description": "一母八子的成套飞刃，锋利无比",\n      "source": "兑换",\n    }\n  ]\n}'
    }
]

def extract_weapons_from_text(text: str, model_name: str = "deepseek-chat") -> WeaponExtractionResult:
    """
    使用DeepSeek API从给定文本中提取武器信息。

    Args:
        text (str): 要分析的《凡人修仙传》文本片段。
        model_name (str): 要使用的DeepSeek模型名称。

    Returns:
        WeaponExtractionResult: 包含提取到的武器信息的Pydantic模型对象。

    Raises:
        Exception: 如果API请求失败或响应无法解析。
    """
    # 构建消息列表
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *EXAMPLES,  # 加入少样本示例
        {"role": "user", "content": text}
    ]

    # 设置API请求头和数据
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,  # 低温度以减少随机性，使输出更确定
        "max_tokens": 4000,  # 根据实际需要调整
        "response_format": { "type": "json_object" }  # 如果DeepSeek API支持，请求JSON输出格式
    }

    try:
        # 发送请求到DeepSeek API
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        response.raise_for_status()  # 如果请求失败则抛出异常

        response_json = response.json()
        # 提取模型回复的内容
        assistant_reply = response_json["choices"][0]["message"]["content"]

        # 尝试从回复中解析JSON
        # 有时模型可能会在JSON前后添加一些解释性文字，这里尝试提取第一个JSON块
        # 这是一种简单的处理方式，根据模型实际输出可能需要进行调整
        start_idx = assistant_reply.find('{')
        end_idx = assistant_reply.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = assistant_reply[start_idx:end_idx]
        else:
            json_str = assistant_reply  # 如果没有找到大括号，则使用整个回复

        weapon_data = json.loads(json_str)
        return WeaponExtractionResult(**weapon_data)

    except requests.exceptions.RequestException as e:
        print(f"API请求错误: {e}")
        raise
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"解析模型响应时出错: {e}")
        print(f"模型原始响应: {assistant_reply}")
        raise

# 4. 使用示例
if __name__ == "__main__":
    # 示例文本：你可以替换成任何《凡人修仙传》的相关段落
    # sample_text = """
    # 韩立最重要的本命法宝是青竹蜂云剑，乃是用珍稀的金雷竹炼制而成，共七十二口，自带辟邪神雷，专克魔功邪术。
    # 他在结丹后便开始炼制，初期仅能驱使数口，元婴后方能驾驭全部，并不断添加庚精、炼晶等材料强化，后期更在金雷竹中重新种剑淬炼，去除杂质，威力大增。
    # 此外，韩立还拥有一件风雷翅，得自风希，由风雷翅炼制，飞行速度极快，是其跑路保命的利器。
    # 在筑基期时，韩立曾用透明丝线法器阴过不少敌人，此丝线几乎无形，弹性与锋利兼备。
    # """
    directory_path = "data/"
    entries = os.listdir(directory_path)

    # 过滤出文件（排除目录）
    files = [entry for entry in entries
             if os.path.isfile(os.path.join(directory_path, entry))]

    all_dict = {}
    index = 0
    for entry in files[:2]:
        # 提取第一个的
        index += 1
        print(f"提取第{index}个文件")
        with open('data/' + entry, 'r', encoding='utf-8') as file:
            sample_text = file.read()
            try:
                result = extract_weapons_from_text(sample_text)
                print("提取成功！")
                print(f"共提取到 {len(result.weapons)} 件武器：")
                for weapon in result.weapons:
                    print(f"- 名称：{weapon.name}")
                    print(f"  类型：{weapon.type}")
                    print(f"  描述：{weapon.description}")
                    print(f"  来源：{weapon.source}")
                    print(f"  状态：{weapon.status}\n")

                data_dict = result.model_dump()
                all_dict.update(data_dict)
            except Exception as e:
                print(f"提取过程中发生错误: {e}")
                # 2.2 使用 json.dumps 序列化字典，并设置 ensure_ascii=False 确保中文正常显示
    json_str_v2 = json.dumps(all_dict, indent=2, ensure_ascii=False)
        # 你也可以将结果保存为JSON文件
    with open('hanli_weapons_extracted.json', 'w', encoding='utf-8') as f:
                # f.write(result.model_dump_json(indent=2, ensure_ascii=False))
        f.write(json_str_v2)
        print("武器信息已保存到 hanli_weapons_extracted.json")

