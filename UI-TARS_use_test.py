import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import json
import os
import copy
from typing import List, Dict, Any
import re


def eval_agentnet(model_path, eval_path, dst, checkpoint_file="eval.checkpoint"):
    start_line = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_line = int(f.read().strip())
        print(f"从第 {start_line} 行开始续传")
    processor = AutoProcessor.from_pretrained(model_path)
    device = torch.device("cuda:3")
    model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)
    count = 0
    with (
        open(eval_path, "r", encoding="utf-8") as f,
        open(dst, "a" if start_line > 0 else "w", encoding="utf-8") as f_e,
    ):
        # 跳过已经处理的行
        for _ in range(start_line):
            next(f)

        for line_num, line in enumerate(f, start_line + 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            item = json.loads(line)
            messages = item["messages"]
            messages.pop()
            for msg in messages:
                if "<image>" in msg["content"]:
                    msg["content"] = [
                        {"type": "image", "url": item["images"][0]},
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
            output_test = processor.decode(outputs[0][inputs["input_ids"].shape[-1] :])
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_test}],
                }
            )
            result = {"messages": messages, "teacher_action": item["teacher_action"]}
            json_line = json.dumps(result, ensure_ascii=False)
            f_e.write(json_line + "\n")
            if "Confidence:5" not in output_test:
                count += 1
                print(output_test)
            # 更新检查点（每处理一行就更新）
            with open(checkpoint_file, "w") as f_check:
                f_check.write(str(line_num))
        # 处理完成后删除检查点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        print("处理完成！")


def eval_kairos():
    device = torch.device(f"cuda:3")

    # model_path='/newdata/xuys10/UI-TARS-1.5-7B'
    model_path = (
        "/newdata/zhouxy/model/trained_models/uitars_lora_sft_agentnet/merged0905"
    )
    eval_path = "/newdata/zhouxy/dataset/AgentNet/social_media/social_media_eval.jsonl"
    dst = "/newdata/zhouxy/model/trained_models/uitars_lora_sft_agentnet/merged0905/eval0905.jsonl"

    count = 0
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)
    eval_list = []
    with open(eval_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    for item in data:
        messages = item["messages"]
        messages.pop()
        for msg in messages:
            if "<image>" in msg["content"]:
                msg["content"] = [
                    {"type": "image", "url": item["images"][0]},
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
        output_test = processor.decode(outputs[0][inputs["input_ids"].shape[-1] :])
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": output_test}]}
        )
        eval_list.append(
            {"messages": messages, "teacher_action": item["teacher_action"]}
        )
        if "Confidence:5" not in output_test:
            count += 1
            print(processor.decode(outputs[0][inputs["input_ids"].shape[-1] :]))

    # prompt_c='''You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action and confidence to complete the task. \n\n## Output Format\n```\nThought: ...\nAction: ...\nConfidence: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1, y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1, y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1, y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n\n## Note\n- Use Chinese in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part. Evaluate your action to be scored, giving it a score from 1 to 5 in 'Confidence' part. A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.\n\n## User Instruction\nCould you help me search UI-TARS?'''
    # prompt="You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n```\nThought: ...\nAction: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1, y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1, y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1, y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n\n## Note\n- Use Chinese in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.\n\n## User Instruction\nCould you help me search UI-TARS?"
    # local_img = Image.open("chrom_test.png")
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             #{"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
    #             {"type": "image","url": "chrom_test.png"},
    #             {"type": "text", "text": prompt_c}
    #         ]
    #     },
    # ]

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(eval_list, f, ensure_ascii=False, indent=2)
    print(f"验证完成！共 {len(eval_list)} 条记录，已保存至 {dst}")
    print(f"置信度不为5的条数：{count}")


if __name__ == "__main__":
    model_path = (
        "/newdata/zhouxy/model/trained_models/uitars_lora_sft_agentnet/merged0911"
    )
    eval_path = "/newdata/zhouxy/dataset/AgentNet/social_media/social_media_eval.jsonl"
    dst = "/newdata/zhouxy/model/trained_models/uitars_lora_sft_agentnet/merged0911/eval0911.jsonl"

    eval_agentnet(model_path, eval_path, dst)
