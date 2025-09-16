import json
import os
import re


def eval_agentnet(eval_path, t=4):
    tt = 0  # tp confidence==5 action==teacher
    tf = 0  # 假阴fn 需要但不执行交互
    ft = 0  # fp confidence<5 不需要但执行交互了
    ff = 0  # tn
    screen_w = 1280
    with open(eval_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            data = json.loads(line)
            student = data["messages"][-1]["content"][0]["text"]
            teacher = data["teacher_action"]

            pattern = r"Thought:(.*?)\nAction:(.*?)\nConfidence:(\d+)"
            m = re.search(pattern, student, re.S)
            if m:
                thought = m.group(1).strip()
                action = m.group(2).strip()
                conf = int(m.group(3))
            s_right = True
            if teacher.startswith("click"):
                if action.startswith("click"):
                    # 坐标和地面实况之间的距离在屏幕宽度的 14%以内，判断为正确
                    pattern = r"<\|box_start\|>\((-?\d+),\s*(-?\d+)\)<\|box_end\|>"
                    m_t = re.search(pattern, teacher)
                    m_s = re.search(pattern, action)
                    if m_t and m_s:
                        x_t, y_t = map(int, m_t.groups())
                        x_s, y_s = map(int, m_s.groups())
                        threshold = screen_w * 0.14  # 14% 宽度对应的像素
                        distance = ((x_t - x_s) ** 2 + (y_t - y_s) ** 2) ** 0.5
                        if distance > threshold:
                            s_right = False

                    else:
                        raise ValueError(f"无法从代码中提取坐标: {teacher} 或 {action}")
                        s_right = False
                else:
                    s_right = False
            elif teacher.startswith("left_double"):
                if action.startswith("left_double"):
                    # 坐标和地面实况之间的距离在屏幕宽度的 14%以内，判断为正确
                    pattern = r"<\|box_start\|>\((-?\d+),\s*(-?\d+)\)<\|box_end\|>"
                    m_t = re.search(pattern, teacher)
                    m_s = re.search(pattern, action)
                    if m_t and m_s:
                        x_t, y_t = map(int, m_t.groups())
                        x_s, y_s = map(int, m_s.groups())
                        threshold = screen_w * 0.14  # 14% 宽度对应的像素
                        distance = ((x_t - x_s) ** 2 + (y_t - y_s) ** 2) ** 0.5
                        if distance > threshold:
                            s_right = False
                    else:
                        raise ValueError(f"无法从代码中提取坐标: {teacher} 或 {action}")
                else:
                    s_right = False
            elif teacher.startswith("right_single"):
                if action.startswith("right_single"):
                    # 坐标和地面实况之间的距离在屏幕宽度的 14%以内，判断为正确
                    pattern = r"<\|box_start\|>\((-?\d+),\s*(-?\d+)\)<\|box_end\|>"
                    m_t = re.search(pattern, teacher)
                    m_s = re.search(pattern, action)
                    if m_t and m_s:
                        x_t, y_t = map(int, m_t.groups())
                        x_s, y_s = map(int, m_s.groups())
                        threshold = screen_w * 0.14  # 14% 宽度对应的像素
                        distance = ((x_t - x_s) ** 2 + (y_t - y_s) ** 2) ** 0.5
                        if distance > threshold:
                            s_right = False
                    else:
                        raise ValueError(f"无法从代码中提取坐标: {teacher} 或 {action}")
                else:
                    s_right = False
            elif teacher.startswith("drag"):
                if action.startswith("drag"):
                    # 坐标和地面实况之间的距离在屏幕宽度的 14%以内，判断为正确
                    pattern = r"box_start\|>\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\).*?box_start\|>\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)"
                    m_t = re.search(pattern, teacher)
                    m_s = re.search(pattern, action)
                    if m_t and m_s:
                        x1_t, y1_t, x2_t, y2_t = map(int, m_t.groups())
                        x1_s, y1_s, x2_s, y2_s = map(int, m_s.groups())
                        threshold = screen_w * 0.14  # 14% 宽度对应的像素
                        distance1 = ((x1_t - x1_s) ** 2 + (y1_t - y1_s) ** 2) ** 0.5
                        distance2 = ((x2_t - x2_s) ** 2 + (y2_t - y2_s) ** 2) ** 0.5
                        if distance1 > threshold or distance2 > threshold:
                            s_right = False
                    else:
                        raise ValueError(f"无法从代码中提取坐标: {teacher} 或 {action}")
                else:
                    s_right = False
            elif teacher.startswith("hotkey"):
                if action.startswith("hotkey"):
                    # 坐标和地面实况之间的距离在屏幕宽度的 14%以内，判断为正确
                    pattern = r"key\s*=\s*['\"](.*?)['\"]"
                    m_t = re.search(pattern, teacher)
                    m_s = re.search(pattern, action)
                    if m_t and m_s:
                        keys_t = m_t.group(1).split()
                        keys_s = m_s.group(1).split()
                        if set(keys_t) != set(keys_s):
                            s_right = False
                    else:
                        raise ValueError(f"无法从代码中提取坐标: {teacher} 或 {action}")
                else:
                    s_right = False
            elif teacher.startswith("type"):
                if action.startswith("type"):
                    m_t = re.search(r"type\(content='(.*)'\)", teacher, re.S)
                    m_s = re.search(r"type\(content='(.*)'\)", action, re.S)
                    if m_s and m_t:
                        text_s = m_s.group(1)
                        text_t = m_t.group(1)
                    else:
                        raise ValueError(f"无法从代码中提取坐标: {teacher} 或 {action}")
                    if text_t not in text_s:
                        s_right = False
                else:
                    s_right = False
            elif teacher.startswith("scroll"):
                if action.startswith("scroll"):
                    pat = re.compile(
                        r"scroll\(.*?start_box\s*=\s*['\"]<\|box_start\|>\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)<\|box_end\|>['\"].*?"
                        r"direction\s*=\s*['\"](.*?)['\"]"
                        r"|"  # 或参数顺序相反
                        r"scroll\(.*?direction\s*=\s*['\"](.*?)['\"].*?"
                        r"start_box\s*=\s*['\"]<\|box_start\|>\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)<\|box_end\|>['\"]"
                    )

                    for m in pat.finditer(action):
                        if m.group(1):  # 第一种顺序
                            x_s, y_s, direction_s = m.group(1), m.group(2), m.group(3)
                        else:  # 第二种顺序
                            direction_s, x_s, y_s = m.group(4), m.group(5), m.group(6)
                    for m in pat.finditer(teacher):
                        if m.group(1):  # 第一种顺序
                            x_t, y_t, direction_t = m.group(1), m.group(2), m.group(3)
                        else:  # 第二种顺序
                            direction_t, x_t, y_t = m.group(4), m.group(5), m.group(6)

                    x_t, y_t = float(x_t), float(y_t)
                    x_s, y_s = float(x_s), float(y_s)
                    threshold = screen_w * 0.14  # 14% 宽度对应的像素
                    distance = ((x_t - x_s) ** 2 + (y_t - y_s) ** 2) ** 0.5
                    if distance > threshold or direction_t != direction_s:
                        s_right = False
                else:
                    s_right = False
            elif teacher.startswith("finished"):
                if not action.startswith("finished"):
                    s_right = False
            else:
                raise ValueError(f"无法识别的代码: {teacher}")
            if s_right:
                # action 正确（与teacher一致）
                if conf > t:
                    tt += 1
                else:
                    ft += 1
            else:
                # action 错误
                if conf > t:
                    tf += 1
                else:
                    ff += 1
    print(f"tt={tt},tf={tf},ft={ft},ff={ff}")

    print(f"准确率HSR={(tt + ff) / (tt + tf + ft + ff)}")
    print(f"IP(需要交互的准确率)={ff / (ff + tf)}")
    print(f"AP(自主准确率)={tt / (tt + ft)}")


if __name__ == "__main__":
    eval_path = "/newdata/zhouxy/model/trained_models/uitars_lora_sft_agentnet/merged0908/eval0908.jsonl"
    eval_agentnet(eval_path, 3)
