import json
from typing import List, Dict, Any
import re

eval_path = "/newdata/zhouxy/model/uitars_lora_merge/teacher/eval828.json"
teacher_path = "/data/zhouxy/kairos/test_teacher828.json"

with open(eval_path, "r", encoding="utf-8") as f:
    data_eval: List[Dict[str, Any]] = json.load(f)

with open(teacher_path, "r", encoding="utf-8") as f:
    data_teacher: List[Dict[str, Any]] = json.load(f)

tt = 0  # tp confidence==5 action==teacher
tf = 0  # 假阴fn 需要但不执行交互
ft = 0  # fp confidence<5 不需要但执行交互了
ff = 0  # tn
screen_w = 720
# for item_e,item_t in zip(data_eval,data_train):
#     eval=item_e['messages'][-1]['content'][0]['text']

#     train=item_t['messages'][-1]['content']
#     if 'Confidence:5' in train:

for item, item_t in zip(data_eval, data_teacher):
    student = item["messages"][-1]["content"][0]["text"]
    pattern = r"Thought:(.*?)\nAction:(.*?)\nConfidence:(\d+)"
    m = re.search(pattern, student, re.S)
    if m:
        thought = m.group(1).strip()
        action = m.group(2).strip()
        conf = int(m.group(3))
    teacher = item_t["teacher_action"]
    s_right = True
    if teacher.startswith("CLICK"):
        if action.startswith("click"):
            # 坐标和地面实况之间的距离在屏幕宽度的 14%以内，判断为正确
            m_t = re.search(r"\[\[(\d+),(\d+)\]\]", teacher)
            if m_t:
                x_t, y_t = map(int, m_t.groups())
                x_t = x_t * screen_w / 1084
                y_t = y_t * screen_w / 1084
            m_s = re.search(
                r"<\|box_start\|>\((-?\d+),\s*(-?\d+)\)<\|box_end\|>", action
            )
            if m_s:
                x_s, y_s = map(int, m_s.groups())
                threshold = screen_w * 0.14  # 14% 宽度对应的像素
                distance = ((x_t - x_s) ** 2 + (y_t - y_s) ** 2) ** 0.5
                if distance > threshold:
                    s_right = False

            else:
                s_right = False
        else:
            s_right = False
    elif teacher.startswith("TYPE"):
        if action.startswith("type"):
            m_t = re.search(r"TYPE \[(.*)\]", teacher)
            if m_t:
                text_t = m_t.group(1)
            m_s = re.search(r"type\(content='(.*)'\)", action, re.S)
            if m_s:
                text_s = m_s.group(1)
            if text_t not in text_s:
                s_right = False
        else:
            s_right = False
    elif teacher.startswith("SCROLL"):
        if action.startswith("scroll"):
            m_t = re.search(r"SCROLL \[(.*)\]", teacher)
            if m_t:
                text_t = m_t.group(1)

            if text_t.lower not in action:
                s_right = False
        else:
            s_right = False
    else:
        continue
    if s_right:
        # action 正确（与teacher一致）
        if conf == 5:
            tt += 1
        else:
            ft += 1
    else:
        # action 错误
        if conf == 5:
            tf += 1
        else:
            ff += 1
print(f"tt={tt},tf={tf},ft={ft},ff={ff}")
print(f"准确率HSR={(tt + ff) / (tt + tf + ft + ff)}")
print(f"IP(需要交互的准确率)={ff / (ff + tf)}")
print(f"AP(自主准确率)={tt / (tt + ft)}")
