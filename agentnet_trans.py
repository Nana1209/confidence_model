import json
import random
import re
import torch
from PIL import Image
import os
from pathlib import Path
import shutil
from transformers import AutoProcessor, AutoModelForVision2Seq

from eval_agentnet import eval_action

# device = torch.device(f"cuda:0")
# model_path = "/newdata/xuys10/UI-TARS-1.5-7B"

# processor = AutoProcessor.from_pretrained(model_path)
# model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)
# path = "/newdata/zhouxy/dataset/AgentNet/social_media/social_media.jsonl"
# image_path = "/newdata/zhouxy/dataset/AgentNet/social_media/images/"
# w_path = "/newdata/zhouxy/dataset/AgentNet/social_media/social_media_uitars.jsonl"


def answer_dataset(path, image_path, w_path, checkpoint_file=".checkpoint"):
    start_line = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_line = int(f.read().strip())
        print(f"从第 {start_line} 行开始续传")

    with (
        open(path, "r", encoding="utf-8") as f,
        open(w_path, "a" if start_line > 0 else "w", encoding="utf-8") as f_w,
    ):
        # 跳过已经处理的行
        for _ in range(start_line):
            next(f)

        for line_num, line in enumerate(f, start_line + 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            data = json.loads(line)
            history = []  # 把assistant的历史回答action累计在这里
            if data["task_difficulty"] < 7:
                actions = data["traj"]
            else:
                actions = data["traj"][:10]
            for action in actions:
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action and confidence to complete the task. \n\n## Output Format\n```\nThought: ...\nAction: ...\nConfidence: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1, y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1, y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1, y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n\n## Note\n- Use Chinese in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part. \n\n## User Instruction\n{data['natural_language_task']}",
                            }
                        ],
                    },
                ]
                if history:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "\n-------------------\n".join(history),
                                }
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "url": f"{image_path + action['image']}",
                                }
                            ],
                        }
                    )
                else:
                    messages[1]["content"].append(
                        {"type": "image", "url": f"{image_path + action['image']}"}
                    )
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                outputs = model.generate(**inputs, max_new_tokens=1024)
                output_text = processor.decode(
                    outputs[0][inputs["input_ids"].shape[-1] :]
                )
                history.append(output_text)
                action["uitars"] = output_text
            json_line = json.dumps(data, ensure_ascii=False)
            f_w.write(json_line + "\n")

            # 更新检查点（每处理一行就更新）
            with open(checkpoint_file, "w") as f_check:
                f_check.write(str(line_num))

    # 处理完成后删除检查点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print("处理完成！")


def process_dataset(
    path, image_path, train_path, eval_path, checkpoint_file="train.checkpoint"
):
    """agentnet生成数据集，历史action放messages的历史对话中"""
    start_line = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_line = int(f.read().strip())
        print(f"从第 {start_line} 行开始续传")
    new_path = "/newdata/zhouxy/dataset/AgentNet/social_media/social_media_confidence_rule.jsonl"
    rule_count = 0
    claude_count = 0
    with (
        open(path, "r", encoding="utf-8") as f,
        open(train_path, "a" if start_line > 0 else "w", encoding="utf-8") as f_t,
        open(eval_path, "a" if start_line > 0 else "w", encoding="utf-8") as f_e,
        open(new_path, "a" if start_line > 0 else "w", encoding="utf-8") as f_new,
    ):
        # 跳过已经处理的行
        for _ in range(start_line):
            next(f)

        for line_num, line in enumerate(f, start_line + 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            data = json.loads(line)
            previous_actions_list = []  # 把assistant的历史回答action累计在这里,用agentnet里的历史action
            if data["task_difficulty"] < 7:
                actions = data["traj"]
            else:
                actions = data["traj"][:10]
            final_goal = data["natural_language_task"]

            for i, action in enumerate(actions):
                confidence = action["confidence"]
                messages = []
                images = [image_path + action["image"]]
                messages.append(
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    }
                )
                if previous_actions_list:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action and confidence to complete the task. \n\n## Output Format\n```\nThought: ...\nAction: ...\nConfidence: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1, y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1, y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1, y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n\n## Note\n- Use Chinese in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part. Evaluate your action to be scored, giving it a score from 1 to 5 in 'Confidence' part. A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.\n\n## User Instruction\n{final_goal}",
                        }
                    )
                    n = min(1, len(previous_actions_list))  # 添加历史对话
                    previous_actions = "\n-------------".join(
                        previous_actions_list[-n:]
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"{previous_actions}",
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": "<image>",
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action and confidence to complete the task. \n\n## Output Format\n```\nThought: ...\nAction: ...\nConfidence: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1, y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1, y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1, y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n\n## Note\n- Use Chinese in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part. Evaluate your action to be scored, giving it a score from 1 to 5 in 'Confidence' part. A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.\n\n## User Instruction\n{final_goal}<image>",
                        }
                    )
                teacher_action = (
                    action["value"]["code_uitars"]
                    if action["value"]["last_step_correct"]
                    else action["claude"]
                )
                # pattern = r"Thought:(.*?)\nAction:(.*?)"
                # m = re.search(pattern, action["uitars"], re.S)
                m = re.search(
                    r"Thought:\s*(.*?)\s+Action:\s*(.*?)(?:<\|im_end\|>|$)",
                    action["uitars"],
                    re.S,
                )
                if m:
                    thought = m.group(1).strip()
                    stu_action = m.group(2).strip()
                else:
                    continue
                if stu_action == "":
                    raise ValueError(f"无法识别的代码: {action['uitars']}")
                _, confidence_rule = eval_action(stu_action, teacher_action, 1280)
                action["confidence_rule"] = confidence_rule

                # assistant = f"{action['uitars']}\nConfidence:{confidence}" #claude生成的confidence
                assistant = (
                    f"{action['uitars']}\nConfidence:{confidence_rule}"
                    if confidence_rule > confidence
                    else f"{action['uitars']}\nConfidence:{confidence}"
                )  # 规则生成的confidence
                if confidence_rule < 5:
                    rule_count += 1
                if confidence < 5:
                    claude_count += 1
                previous_actions_list.append(assistant)
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant,
                    }
                )

                jsondata = {
                    "messages": messages,
                    "images": images,
                    "teacher_action": teacher_action,
                    "id": f"{data['task_id']}+{action['index']}",
                }
                json_line = json.dumps(jsondata, ensure_ascii=False)
                if random.random() < 0.80:
                    f_t.write(json_line + "\n")
                else:
                    f_e.write(json_line + "\n")
                # 非5分的把teacher——action作为5分数据加入数据集
                if confidence_rule < 5:
                    messages.pop()
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"Thought:{action['value']['thought']}\nAction:{teacher_action}\nConfidence:5",
                        }
                    )
                    jsondata = {
                        "messages": messages,
                        "images": images,
                        "teacher_action": teacher_action,
                        "id": f"{data['task_id']}+{action['index']}",
                    }
                    json_line = json.dumps(jsondata, ensure_ascii=False)
                    if random.random() < 0.80:
                        f_t.write(json_line + "\n")
                    else:
                        f_e.write(json_line + "\n")

            json_new = json.dumps(data, ensure_ascii=False)
            f_new.write(json_new + "\n")
            # 更新检查点（每处理一行就更新）
            with open(checkpoint_file, "w") as f_check:
                f_check.write(str(line_num))

    # 处理完成后删除检查点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print("处理完成！")
    print("rule_count:", rule_count)
    print("claude_count:", claude_count)


def gen_dataset(
    path, image_path, train_path, eval_path, checkpoint_file="train.checkpoint"
):
    """生成训练集和验证集，历史action放user prompt中"""
    prompt = "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action and confidence to complete the task. \n\n## Output Format\n```\nThought: ...\nAction: ...\nConfidence: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1, y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1, y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1, y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n\n## Note\n- Use Chinese in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part. Evaluate your action to be scored, giving it a score from 1 to 5 in 'Confidence' part. A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.\n\n## User Instruction\n{final_goal}\n\n## Previous Actions\n{previous_actions}\n<image>"

    start_line = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_line = int(f.read().strip())
        print(f"从第 {start_line} 行开始续传")
    rule_count = 0
    claude_count = 0
    with (
        open(path, "r", encoding="utf-8") as f,
        open(train_path, "a" if start_line > 0 else "w", encoding="utf-8") as f_t,
        open(eval_path, "a" if start_line > 0 else "w", encoding="utf-8") as f_e,
    ):
        # 跳过已经处理的行
        for _ in range(start_line):
            next(f)

        for line_num, line in enumerate(f, start_line + 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            data = json.loads(line)
            previous_actions_list = []  # agentnet里的历史动作，当作pre
            if data["task_difficulty"] < 7:
                actions = data["traj"]
            else:
                actions = data["traj"][:10]
            final_goal = data["natural_language_task"]

            for i, action in enumerate(actions):
                confidence = action["confidence"]
                messages = []
                images = [image_path + action["image"]]
                messages.append(
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    }
                )
                if previous_actions_list:
                    n = min(5, len(previous_actions_list))  # 添加历史对话
                    previous_actions = "\n".join(previous_actions_list[-n:])
                    messages.append(
                        {
                            "role": "user",
                            "content": prompt.format(
                                final_goal=final_goal, previous_actions=previous_actions
                            ),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": prompt.format(
                                final_goal=final_goal, previous_actions="None"
                            ),
                        }
                    )
                teacher_action = (
                    action["value"]["code_uitars"]
                    if action["value"]["last_step_correct"]
                    else action["claude"]
                )
                # pattern = r"Thought:(.*?)\nAction:(.*?)"
                # m = re.search(pattern, action["uitars"], re.S)
                m = re.search(
                    r"Thought:\s*(.*?)\s+Action:\s*(.*?)(?:<\|im_end\|>|$)",
                    action["uitars"],
                    re.S,
                )
                if m:
                    thought = m.group(1).strip()
                    stu_action = m.group(2).strip()
                else:
                    continue
                if stu_action == "":
                    raise ValueError(f"无法识别的代码: {action['uitars']}")
                confidence_rule = action["confidence_rule"]

                # assistant = f"{action['uitars']}\nConfidence:{confidence}" #claude生成的confidence
                assistant = (
                    f"{action['uitars']}\nConfidence:{confidence_rule}"
                    if confidence_rule > confidence
                    else f"{action['uitars']}\nConfidence:{confidence}"
                )  # 规则生成的confidence
                if confidence_rule < 5:
                    rule_count += 1
                if confidence < 5:
                    claude_count += 1
                previous_actions_list.append(action["value"]["action"])
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant,
                    }
                )

                jsondata = {
                    "messages": messages,
                    "images": images,
                    "teacher_action": teacher_action,
                    "id": f"{data['task_id']}+{action['index']}",
                }
                json_line = json.dumps(jsondata, ensure_ascii=False)
                if random.random() < 0.80:
                    f_t.write(json_line + "\n")
                else:
                    f_e.write(json_line + "\n")
                # 非5分的把teacher——action作为5分数据加入数据集
                if confidence_rule < 5:
                    messages.pop()
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"Thought:{action['value']['thought']}\nAction:{teacher_action}\nConfidence:5",
                        }
                    )
                    jsondata = {
                        "messages": messages,
                        "images": images,
                        "teacher_action": teacher_action,
                        "id": f"{data['task_id']}+{action['index']}",
                    }
                    json_line = json.dumps(jsondata, ensure_ascii=False)
                    if random.random() < 0.80:
                        f_t.write(json_line + "\n")
                    else:
                        f_e.write(json_line + "\n")

            # 更新检查点（每处理一行就更新）
            with open(checkpoint_file, "w") as f_check:
                f_check.write(str(line_num))

    # 处理完成后删除检查点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print("处理完成！")
    print("rule_count:", rule_count)
    print("claude_count:", claude_count)


def view_dataset():
    """
    action转换：
    pyautogui.click(x=0.3861, y=0.5042)=》click(start_box='<|box_start|>(x1, y1)<|box_end|>')
    pyautogui.hotkey(['cmd', 'c'])=>hotkey(key='')
    pyautogui.moveTo(x=0.267, y=0.5831)\npyautogui.dragTo(x=0.1196, y=0.5838, button='left')=>drag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x2, y2)<|box_end|>')
    pyautogui.write(message='youtube.com')\npyautogui.press('enter')=》type(content='youtube.com\n')
    """

    """
    数据集字段解析：
    instruction：祈使句的任务描述
    natural_language_task：疑问句的任务描述
    actual_task:任务分解
    traj：任务实现action列表，可拆做数据
        action_list：任务实现action列表last_step_correct为False时，表示这步的操作不对
    """
    dst = Path("/newdata/zhouxy/dataset/AgentNet/social_media/images/")
    domain = set()
    dataset_search = []
    dataset_buy = []
    dataset_dld = []
    dataset_excel = []
    dataset_amazon = []
    dataset_word = []
    dataset_fb = []
    easy = 0
    path = "/data/zhouxy/dataset/AgentNet/agentnet_win_mac_18k.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            data = json.loads(line)
            instruction = data["instruction"]
            if "Search" in instruction:
                dataset_search.append(data)
            if "buy" in instruction or "Buy" in instruction:
                dataset_buy.append(data)
            if "download" in instruction or "Download" in instruction:
                dataset_dld.append(data)

            if "Excel" in instruction or "excel" in instruction:
                dataset_excel.append(data)

            if "Ebay" in instruction or "ebay" in instruction:
                dataset_amazon.append(data)
            if "word" in instruction or "Word" in instruction:
                dataset_word.append(data)
            if (
                "facebook" in instruction
                or "Facebook" in instruction
                or "twitter" in instruction
                or "Twitter" in instruction
            ):
                dataset_fb.append(data)
                for action in data["traj"]:
                    imagepath = action["image"]
                    if "_" in imagepath:
                        imagepath = imagepath.split("_")[1]
                    imagepath = (
                        f"/data/zhouxy/dataset/AgentNet/win_mac_images/{imagepath}"
                    )
                    dst.mkdir(parents=True, exist_ok=True)  # 若目录不存在则创建
                    shutil.copy(imagepath, dst)
                if data["task_difficulty"] < 7:
                    easy += 1
            # if "domain" in data:
            #     domain.add(data['domain'])

            # if os.path.exists(imagepath):
            #     print("存在")
            # else:
            #     print("图片不存在")
            # Image.open(imagepath).show()
    print(len(dataset_search))
    print(len(dataset_buy))
    print(len(dataset_dld))
    print(len(dataset_excel))
    print(len(dataset_amazon))
    print(len(dataset_word))
    print(len(dataset_fb))
    print(easy)
    print(domain)  # data 就是一个 Python dict
    Path("/newdata/zhouxy/dataset/AgentNet/social_media/social_media.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in dataset_fb) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    # view_dataset()
    # answer_dataset(path, image_path, w_path)
    image_path = "/newdata/zhouxy/dataset/AgentNet/social_media/images/"
    path = "/newdata/zhouxy/dataset/AgentNet/social_media/social_media_confidence_rule.jsonl"
    train_path = (
        "/newdata/zhouxy/dataset/AgentNet/social_media/social_media_train_pre.jsonl"
    )
    eval_path = (
        "/newdata/zhouxy/dataset/AgentNet/social_media/social_media_eval_pre.jsonl"
    )
    print("开始处理数据集...")
    gen_dataset(path, image_path, train_path, eval_path)
