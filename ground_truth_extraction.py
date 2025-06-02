import pandas as pd
import openai
import time
import json
from tqdm import tqdm

# 配置区
csv_file = 'data/real_world/flight_1k.csv'  # 改成你的文件名
columns_to_convert = ['scheduled_dept','actual_dept', 'scheduled_arrival','actual_arrival']  # 改成你的4列名
output_file = 'data/real_world/flight_1k_withgt.csv'
cache_file = 'llm_cache.json'

# openai>=1.0 新版用法
client = openai.OpenAI(YOUR_API_KEY)
model_name = 'gpt-3.5-turbo'  # 或 'gpt-4o'

# 读取缓存
try:
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    print(f'加载缓存成功，已有 {len(cache)} 条缓存记录')
except FileNotFoundError:
    cache = {}
    print('未找到缓存文件，创建新的缓存')

# 定义调用 LLM 转换函数
def convert_with_llm(text):
    if pd.isna(text) or str(text).strip() == '':
        return ''
    
    text_str = str(text).strip()
    
    # 查缓存
    if text_str in cache:
        return cache[text_str]
    
    # 构造 prompt
    prompt = f"""
请将以下日期时间转换为标准格式：YYYY-MM-DD hh:mm:ss 。

转换规则：
1. 如果原文中没有出现年份，请使用年份 2000，日期设为 2000-01-01。
2. 如果原文是三段斜杠分隔的格式（例如 12/10/11），请按 "月/日/年" (MM/DD/YY) 理解，其中年份为 2 位数字时，请补全为 4 位年份（例如 11 → 2011）。
3. 只返回转换后的结果，不要添加解释或其它内容。

日期时间原文：{text_str}
"""

    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        
        # 存入缓存
        cache[text_str] = result
        return result
    
    except Exception as e:
        print(f'转换失败: {text_str} -> {e}')
        return ''

# 读取 CSV
df = pd.read_csv(csv_file)

# 遍历列
for col in columns_to_convert:
    print(f'\n=== 开始转换列: {col} ===')
    converted_values = []
    
    for value in tqdm(df[col], desc=f'转换 {col}'):
        converted = convert_with_llm(value)
        converted_values.append(converted)
        time.sleep(0.5)  # 防止 API 超速率，0.3-0.5 秒比较稳
    
    # 写入新列
    df[col + '_converted'] = converted_values
    
    # 保存缓存
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# 保存最终结果
df.to_csv(output_file, index=False)
print(f'\n全部转换完成，结果保存为 {output_file}')
