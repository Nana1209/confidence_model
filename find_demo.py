import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import os
import json
import random
import re


def process_dataset(path, image_path, model_path, checkpoint_file="train.checkpoint"):
    processor = AutoProcessor.from_pretrained(model_path)
    device = torch.device("cuda:3")
    model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)

    start_line = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_line = int(f.read().strip())
        print(f"从第 {start_line} 行开始续传")
    rule_count = 0
    claude_count = 0
    with (
        open(path, "r", encoding="utf-8") as f,
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
                    n = min(5, len(previous_actions_list))
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

                for msg in messages:
                    if "<image>" in msg["content"]:
                        msg["content"] = [
                            {"type": "image", "url": image_path + action["image"]},
                            {"type": "text", "text": msg["content"]},
                        ]
                    else:
                        msg["content"] = [{"type": "text", "text": msg["content"]}]
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                outputs = model.generate(**inputs, max_new_tokens=1024)
                output_test = processor.decode(
                    outputs[0][inputs["input_ids"].shape[-1] :]
                )

                print(output_test)

                jsondata = {
                    "messages": messages,
                    "images": images,
                    "teacher_action": teacher_action,
                    "id": f"{data['task_id']}+{action['index']}",
                }
                json_line = json.dumps(jsondata, ensure_ascii=False)

            print("-------------------------------")
            json_new = json.dumps(data, ensure_ascii=False)
            # 更新检查点（每处理一行就更新）
            with open(checkpoint_file, "w") as f_check:
                f_check.write(str(line_num))

    # 处理完成后删除检查点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print("处理完成！")
    print("rule_count:", rule_count)
    print("claude_count:", claude_count)


if __name__ == "__main__":
    model_path = (
        "/newdata/zhouxy/model/trained_models/uitars_lora_sft_agentnet/merged0911"
    )
    image_path = "/newdata/zhouxy/dataset/AgentNet/social_media/images/"
    path = "/newdata/zhouxy/dataset/AgentNet/social_media/social_media_confidence_rule.jsonl"

    print("开始处理数据集...")
    process_dataset(path, image_path, model_path)
