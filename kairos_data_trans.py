# convert_google_json.py
import json
import os
import copy
from typing import List, Dict, Any
import re

# SRC_PATH = r"/home/gpu92/zhouxy/confidence-agent/data/kairos/google/google_data.json"
# DST_PATH = r"/home/gpu92/zhouxy/confidence-agent/data/kairos/google/google_data_openai.json"

# SRC_PATH = r"/home/gpu92/zhouxy/confidence-agent/data/kairos/xiaohongshu/xiaohongshu_data.json"
# DST_PATH = r"/home/gpu92/zhouxy/confidence-agent/data/kairos/xiaohongshu/xiaohongshu_data_openai.json"
# image_path='/home/gpu92/zhouxy/confidence-agent/data/kairos/xiaohongshu/images/'

task = {}
dataset = []
import os
from PIL import Image

# -------------------- 参数区 --------------------
IMAGE_ROOT_DIR = r"/data/zhouxy/kairos/xiaohongshu"  # 要遍历的根目录
image_path = "/data/zhouxy/kairos/xiaohongshu/images/"
# starter_image = "/data/zhouxy/kairos/wechat/images/1736874029.8359137_0.png"
# starter_image = "/data/zhouxy/kairos/Amap/images/1736275897.781981_0.png"
# starter_image = "/data/zhouxy/kairos/amazon/images/1735292392.2579656_0.png"
# starter_image = "/data/zhouxy/kairos/cloud_music/images/1736614680.6518524_0.png"
# starter_image = "/data/zhouxy/kairos/google/images/1736715279.8863294_0.png"
starter_image = "/data/zhouxy/kairos/google_maps/images/1736941825.3687358_0.png"
# image1_path = "/data/zhouxy/kairos/wechat/images/1736874029.8359137_1.png"
# image1_path = "/data/zhouxy/kairos/Amap/images/1736281755.1186986_1.png"
# image1_path = "/data/zhouxy/kairos/amazon/images/1735292392.2579656_1.png"
# image1_path = "/data/zhouxy/kairos/bilibili/images/1735812552.2515638_1.png"
# image1_path = "/data/zhouxy/kairos/cloud_music/images/1736614680.6518524_1.png"
# image1_path = "/data/zhouxy/kairos/douyin/images/1735046860.5258026_1.png"
# image1_path = "/data/zhouxy/kairos/google/images/1736588327.298731_1.png"
# image1_path = "/data/zhouxy/kairos/google_maps/images/1736941825.3687358_1.png"
# image1_path = "/data/zhouxy/kairos/meituan/images/1737016627.9341893_1.png"
# image1_path = "/data/zhouxy/kairos/wechat/images/1736874029.8359137_1.png"
image1_path = "/data/zhouxy/kairos/xiaohongshu/images/1733923577.6856332_1.png"
image2_path = None
SRC_PATH = r"/data/zhouxy/kairos/xiaohongshu/xiaohongshu_data.json"
DST_PATH = r"/data/zhouxy/kairos/xiaohongshu/xiaohongshu_data_openai828.json"


JSON_ROOT_DIR = r"/data/zhouxy/kairos/wechat"  # 需要转换坐标的目录
RATIO_NUM = 720
RATIO_DEN = 1084

TARGET_WIDTH = 720  # 目标宽度
QUALITY = 92  # 保存质量，JPEG/WebP 有效
SKIP_SMALLER = True  # True 表示宽度<=720 的文件跳过
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def merge():
    # 1) 配置
    root_dir = Path("/data/zhouxy/kairos/train_data")  # 待遍历的根目录
    train_path = Path("/data/zhouxy/kairos/train.json")  # 输出文件
    test_path = Path("/data/zhouxy/kairos/test.json")
    ratio = 0.3  # 测试集占比 30%

    # 2) 收集所有 .json 文件并合并
    all_data = []
    for json_file in root_dir.rglob("*.json"):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):  # 只处理 list 结构
                all_data.extend(data)
        except Exception as e:
            print(f"跳过文件 {json_file}: {e}")

    # 3) 随机划分
    random.seed(42)  # 固定随机种子，可复现
    random.shuffle(all_data)
    split_idx = int(len(all_data) * ratio)
    test_data = all_data[:split_idx]
    train_data = all_data[split_idx:]

    # 4) 保存
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with test_path.open("w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(
        f"合并完成！总条数 {len(all_data)}，训练 {len(train_data)}，测试 {len(test_data)}"
    )


def merge_teacher():
    import json
    from pathlib import Path
    import random

    root = Path("/data/zhouxy/kairos")
    train_path = Path("/data/zhouxy/kairos/train_teacher.json")  # 输出文件
    test_path = Path("/data/zhouxy/kairos/test_teacher.json")
    ratio = 0.3
    all_data = []
    for sub_dir in root.iterdir():
        if not sub_dir.is_dir():
            continue

        openai_files = list(sub_dir.glob("*openai1.json"))
        data_files = list(sub_dir.glob("*data.json"))

        if openai_files and data_files:
            with (
                open(openai_files[0], "r", encoding="utf-8") as fo,
                open(data_files[0], "r", encoding="utf-8") as fd,
            ):
                o_list = json.load(fo)
                d_list = json.load(fd)

            for o, d in zip(o_list, d_list):
                o["teacher"] = d["teacher_action"]
            all_data.extend(o_list)

    random.seed(42)  # 固定随机种子，可复现
    random.shuffle(all_data)
    split_idx = int(len(all_data) * ratio)
    test_data = all_data[:split_idx]
    train_data = all_data[split_idx:]

    # 4) 保存
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with test_path.open("w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(
        f"合并完成！总条数 {len(all_data)}，训练 {len(train_data)}，测试 {len(test_data)}"
    )


def convert_single(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    {
        "task": "Search the distance from Earth to Mars on Google.",
        "image_path": "/data1/wuzh/google/images/1736537061.8294396_1.png",
        "list": [
            " Open Google.  ",
            " Click on the search box on the screen.  ",
            " Type \"distance from Earth to Mars.\"  ",
            " Select the correct search result or press Enter.  "
        ],
        "now_step": 0,
        "previous_actions": [
            "CLICK <point>[[858,367]]</point>"
        ],
        "score": 1,
        "osatlas_action": "CLICK <point>[[424,274]]</point>",
        "teacher_action": "CLICK <point>[[305,270]]</point>",
        "success": false
    }
    """
    # 转换action
    action = item["osatlas_action"]
    new_action = ""
    if action.startswith("CLICK"):
        start = action.find("[[") + 2
        end = action.find("]]")
        point = action[start:end]
        new_action = f"click(start_box='<|box_start|>({point})<|box_end|>')"
    elif action.startswith("TYPE"):
        start = action.find("[") + 1
        end = action.find("]")
        content = action[start:end]
        new_action = f"type(content='{content}\n')"
    else:
        print("error")
    now_step = item["now_step"]
    new_item = {}
    # 还原任务
    if item["task"] in task:
        new_item = copy.deepcopy(task[item["task"]])
        new_item["messages"].append({"role": "user", "content": "<image>"})
        new_item["messages"].append(
            {
                "role": "assistant",
                "content": "Thought:"
                + item["list"][now_step + 1]
                + "\nAction:"
                + new_action
                + "\nConfidence:"
                + str(item["score"]),
            }
        )
        if "_1" in item["image_path"]:
            new_item["images"].append(image_path + "1736588327.298731_1.png")
        else:
            new_item["images"].append(image_path + os.path.basename(item["image_path"]))
    else:
        messages = []
        images = []

        messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append(
            {
                "role": "user",
                "content": "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action and confidence to complete the task. \n\n## Output Format\n```\nThought: ...\nAction: ...\nConfidence: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1, y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1, y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1, y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n\n## Note\n- Use Chinese in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part. Evaluate your action to be scored, giving it a score from 1 to 5 in 'Confidence' part. A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.\n\n## User Instruction\n"
                + item["task"]
                + "<image>",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": "Thought:"
                + item["list"][0]
                + "\nAction:"
                + new_action
                + "\nConfidence:"
                + str(item["score"]),
            }
        )
        images.append(image_path + "1736715279.8863294_0.png")
        new_item = {"messages": messages, "images": images}
    task[item["task"]] = new_item
    # dataset.append(new_item)

    return new_item


def trans_action(action):
    if action.startswith("CLICK"):
        start = action.find("[[") + 2
        end = action.find("]]")
        point = action[start:end]
        new_action = f"click(start_box='<|box_start|>({point})<|box_end|>')"
    elif action.startswith("TYPE"):
        start = action.find("[") + 1
        end = action.find("]")
        content = action[start:end]
        new_action = f"type(content='{content}\n')"
    elif action.startswith("SCROLL"):
        start = action.find("[") + 1
        end = action.find("]")
        new_action = f"scroll(start_box='<|box_start|>(360,800)<|box_end|>', direction='{action[start:end].lower()}')"
    elif action.startswith("PRESS"):
        new_action = "hotkey(key='home')"
    elif action.startswith("COMPLETE"):
        new_action = "finished()"
    elif action.startswith("IMPOSSIBLE"):
        return None
    else:
        print("转换action error")
    return new_action


def convert_single_new(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    {
        "task": "Search the distance from Earth to Mars on Google.",
        "image_path": "/data1/wuzh/google/images/1736537061.8294396_1.png",
        "list": [
            " Open Google.  ",
            " Click on the search box on the screen.  ",
            " Type \"distance from Earth to Mars.\"  ",
            " Select the correct search result or press Enter.  "
        ],
        "now_step": 0,
        "previous_actions": [
            "CLICK <point>[[858,367]]</point>"
        ],
        "score": 1,
        "osatlas_action": "CLICK <point>[[424,274]]</point>",
        "teacher_action": "CLICK <point>[[305,270]]</point>",
        "success": false
    }
    """
    # 转换action
    action = item["osatlas_action"]
    teacher = item["teacher_action"]

    new_action = trans_action(action)
    if new_action == None:
        return None
    now_step = item["now_step"]
    new_item = {}
    messages = []
    images = []
    if len(item["previous_actions"]) == 0:
        messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append(
            {
                "role": "user",
                "content": "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action and confidence to complete the task. \n\n## Output Format\n```\nThought: ...\nAction: ...\nConfidence: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1, y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1, y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1, y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n\n## Note\n- Use Chinese in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part. Evaluate your action to be scored, giving it a score from 1 to 5 in 'Confidence' part. A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.\n\n## User Instruction\n"
                + item["task"]
                + "<image>",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": "Thought:"
                + item["list"][0]
                + "\nAction:"
                + new_action
                + "\nConfidence:"
                + str(item["score"]),
            }
        )

        full_path = os.path.join(image_path, os.path.basename(item["image_path"]))
        if os.path.isfile(full_path):
            images.append(full_path)
        else:
            images.append(starter_image)
            # full_path = os.path.join(image_path, "1734057910.3515482_0.png")
            print(f"图片不存在: {full_path}")
    else:
        messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append(
            {
                "role": "user",
                "content": "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action and confidence to complete the task. \n\n## Output Format\n```\nThought: ...\nAction: ...\nConfidence: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1, y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1, y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1, y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n\n## Note\n- Use Chinese in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part. Evaluate your action to be scored, giving it a score from 1 to 5 in 'Confidence' part. A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.\n\n## User Instruction\n"
                + item["task"],
            }
        )
        for i, pre_action in enumerate(item["previous_actions"]):
            messages.append(
                {
                    "role": "assistant",
                    "content": "Thought:"
                    + item["list"][i]
                    + "\nAction:"
                    + trans_action(pre_action)
                    + "\nConfidence:5",
                }
            )
        messages.append({"role": "user", "content": "<image>"})
        messages.append(
            {
                "role": "assistant",
                "content": "Thought:"
                + item["list"][-1]
                + "\nAction:"
                + new_action
                + "\nConfidence:"
                + str(item["score"]),
            }
        )
        full_path = os.path.join(image_path, os.path.basename(item["image_path"]))

        if os.path.isfile(full_path):
            images.append(full_path)
        else:
            if "_1" in full_path:
                images.append(image1_path)
            elif "_2" in full_path:
                if image2_path is None:
                    print(f"图片不存在: {full_path}")
                    return None
                images.append(image2_path)
            else:
                print(f"图片不存在: {full_path}")
                return None

    new_item = {"messages": messages, "images": images, "teacher_action": teacher}

    return new_item


def convert_single_1image(item: Dict[str, Any]) -> Dict[str, Any]:
    """
     {
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action and confidence to complete the task. \n\n## Output Format\n```\nThought: ...\nAction: ...\nConfidence: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1, y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1, y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1, y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n\n## Note\n- Use Chinese in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part. Evaluate your action to be scored, giving it a score from 1 to 5 in 'Confidence' part. A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.\n\n## User Instruction\nPlease use Google to search for updates on the latest AI models.<image>"
        },
        {
          "role": "assistant",
          "content": "Thought: Open Google.  \nAction:click(start_box='<|box_start|>(858,367)<|box_end|>')\nConfidence:5"
        },
        {
          "role": "user",
          "content": "<image>"
        },
        {
          "role": "assistant",
          "content": "Thought: Click on the search box on the screen.  \nAction:click(start_box='<|box_start|>(428,274)<|box_end|>')\nConfidence:2"
        }
      ],
      "images": [
        "/home/gpu92/zhouxy/confidence-agent/data/kairos/google/images/1736715279.8863294_0.png",
        "/home/gpu92/zhouxy/confidence-agent/data/kairos/google/images/1736588327.298731_1.png"
      ]
    },
    """
    last_user = None
    last_assistant = None
    new_messages = []
    new_images = []
    if len(item["messages"]) > 3:
        for index, message in enumerate(item["messages"]):
            if index < 3:
                new_messages.append(message)
            else:
                if message["role"] == "user":
                    last_user = message
                elif message["role"] == "assistant":
                    last_assistant = message
                    new_messages.append(last_assistant)
        del new_messages[-1]
        new_messages.append(last_user)
        new_messages.append(last_assistant)
    else:
        return item
    new_images.append(item["images"][-1])
    new_item = {
        "images": new_images,
        "messages": new_messages,
    }

    return new_item


def convert_file(src: str, dst: str) -> None:
    if not os.path.isfile(src):
        raise FileNotFoundError(src)

    with open(src, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # converted = [convert_single(item) for item in data]
    converted = [
        x for x in (convert_single_new(item) for item in data) if x is not None
    ]

    # 如果目录不存在自动创建
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"转换完成！共 {len(converted)} 条记录，已保存至 {dst}")


def convert_file_teacher():
    import json
    from pathlib import Path
    import random

    root = Path("/data/zhouxy/kairos")
    train_path = Path("/data/zhouxy/kairos/train_teacher828.json")  # 输出文件
    test_path = Path("/data/zhouxy/kairos/test_teacher828.json")
    ratio = 0.3
    all_data = []
    for sub_dir in root.iterdir():
        if not sub_dir.is_dir():
            continue

        # openai_files = list(sub_dir.glob("*openai1.json"))
        data_files = list(sub_dir.glob("*828.json"))

        if data_files:
            with open(data_files[0], "r", encoding="utf-8") as fd:
                d_list = json.load(fd)

            all_data.extend(d_list)

    random.seed(42)  # 固定随机种子，可复现
    random.shuffle(all_data)
    split_idx = int(len(all_data) * ratio)
    test_data = all_data[:split_idx]
    train_data = all_data[split_idx:]

    # 4) 保存
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with test_path.open("w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(
        f"合并完成！总条数 {len(all_data)}，训练 {len(train_data)}，测试 {len(test_data)}"
    )


def merge_assistant(src: str, dst: str) -> None:
    if not os.path.isfile(src):
        raise FileNotFoundError(src)

    with open(src, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # converted = [convert_single(item) for item in data]
    converted = [
        x for x in (merge_assistant_single(item) for item in data) if x is not None
    ]

    # 如果目录不存在自动创建
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"转换完成！共 {len(converted)} 条记录，已保存至 {dst}")


def merge_assistant_single(item: Dict[str, Any]) -> Dict[str, Any]:
    lst_msg = None
    new_messages = []
    for msg in item["messages"]:
        if (
            lst_msg is not None
            and lst_msg["role"] == "assistant"
            and msg["role"] == "assistant"
        ):
            ass = new_messages.pop()
            ass["content"] = (
                ass["content"] + "\n-------------------------" + msg["content"]
            )
            new_messages.append(ass)
        else:
            new_messages.append(msg)
            lst_msg = msg
    item["messages"] = new_messages
    return item


def resize_image(file_path):
    """对单张图片进行等比例缩放"""
    try:
        with Image.open(file_path) as img:
            if img.mode in ("RGBA", "LA", "P"):
                # RGBA/LA/P 先转 RGB，避免 JPEG 保存出错，如需保留透明通道可改逻辑
                img = img.convert("RGB")

            if SKIP_SMALLER and img.width <= TARGET_WIDTH:
                print(f"[SKIP] {file_path} 宽度 {img.width} ≤ {TARGET_WIDTH}")
                return

            # 计算等比例高度
            new_width = TARGET_WIDTH
            new_height = int(img.height * new_width / img.width)

            resized = img.resize((new_width, new_height), Image.LANCZOS)

            # 按原格式保存；若原格式为 WebP 但保存为 .jpg，可改成强制 ".jpg"
            resized.save(file_path, quality=QUALITY, optimize=True)
            print(
                f"[OK]   {file_path}  {img.width}×{img.height} → {new_width}×{new_height}"
            )
    except Exception as e:
        print(f"[ERR]  {file_path}  {e}")


def traverse_and_resize(root_dir):
    """递归遍历目录"""
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in ALLOWED_EXTS:
                full_path = os.path.join(dirpath, fname)
                resize_image(full_path)


# 正则：捕获 <|box_start|>(x,y)<|box_end|>
pattern = re.compile(r"<\|box_start\|>\((-?\d+),\s*(-?\d+)\)<\|box_end\|>")


def rescale_xy(m):
    """正则替换回调：把 (x,y) → (x*720/1084, y*720/1084) 取整"""
    x, y = map(int, m.groups())
    new_x = x * RATIO_NUM // RATIO_DEN
    new_y = y * RATIO_NUM // RATIO_DEN
    return f"<|box_start|>({new_x},{new_y})<|box_end|>"


def process_file(path):
    """处理单个 json：字符串整体替换后写回"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    new_text, n = pattern.subn(rescale_xy, text)
    if n == 0:  # 没匹配到任何坐标
        return

    # 写回原地覆盖
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"[UPD] {path} 替换了 {n} 处坐标")


def traverse_xy(root):
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if fname.lower().endswith(".json"):
                process_file(os.path.join(dirpath, fname))


if __name__ == "__main__":
    # convert_file(SRC_PATH, DST_PATH)
    # traverse_and_resize(IMAGE_ROOT_DIR)
    # traverse_xy(JSON_ROOT_DIR)

    src = "/data/zhouxy/kairos/xiaohongshu/xiaohongshu_data_openai.json"
    dst = "/data/zhouxy/kairos/xiaohongshu/xiaohongshu_data_openai1.json"
    # merge_assistant(src, dst)
    convert_file_teacher()
    print("全部处理完成！")
