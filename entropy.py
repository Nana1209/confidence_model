import json
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForVision2Seq

import h5py
import torch
from tqdm import tqdm
import numpy as np
import os


class HDF5OutputSaver:
    def __init__(self, hdf5_filename, mode="a"):
        """
        初始化HDF5保存器

        Args:
            hdf5_filename: HDF5文件名
            mode: 文件模式 'a'=追加, 'w'=覆盖
        """
        self.hdf5_filename = hdf5_filename
        self.mode = mode

        # 检查文件是否存在
        file_exists = os.path.exists(hdf5_filename)

        # 打开文件
        self.file = h5py.File(hdf5_filename, mode)

        # 如果文件不存在或者模式是覆盖，创建组
        if not file_exists or mode == "w":
            if "samples" not in self.file:
                self.file.create_group("samples")
            if "metadata" not in self.file:
                self.file.create_group("metadata")
                self.file["metadata"].attrs["total_samples"] = 0
                self.file["metadata"].attrs["created_time"] = str(np.datetime64("now"))

    def sample_exists(self, sample_id):
        """检查样本是否已存在"""
        return sample_id in self.file["samples"]

    def save_sample(self, sample_id, inputs, outputs, overwrite=False):
        """保存单个样本的outputs"""

        # 检查样本是否已存在
        if self.sample_exists(sample_id) and not overwrite:
            print(f"样本 {sample_id} 已存在，跳过保存")
            return False

        # 如果存在且要覆盖，先删除
        if self.sample_exists(sample_id) and overwrite:
            del self.file["samples"][sample_id]

        sample_group = self.file["samples"].create_group(sample_id)

        # 保存metadata
        sample_group.attrs["sample_id"] = sample_id
        sample_group.attrs["sequence_length"] = outputs.sequences.shape[1]
        sample_group.attrs["num_generation_steps"] = (
            len(outputs.scores) if outputs.scores else 0
        )
        sample_group.attrs["saved_time"] = str(np.datetime64("now"))

        # 保存inputs（可选）
        if "input_ids" in inputs:
            sample_group.create_dataset(
                "input_ids", data=inputs["input_ids"].cpu().numpy()
            )
        if "pixel_values" in inputs:
            sample_group.create_dataset(
                "pixel_values", data=inputs["pixel_values"].cpu().numpy()
            )

        # 保存outputs的sequences
        sample_group.create_dataset("sequences", data=outputs.sequences.cpu().numpy())

        # 保存outputs的scores
        if outputs.scores is not None:
            scores_group = sample_group.create_group("scores")
            for step, score in enumerate(outputs.scores):
                scores_group.create_dataset(
                    f"step_{step:04d}", data=score.cpu().numpy()
                )

        # 更新总样本数
        if "metadata" in self.file:
            current_count = self.file["metadata"].attrs.get("total_samples", 0)
            self.file["metadata"].attrs["total_samples"] = current_count + 1
            self.file["metadata"].attrs["last_updated"] = str(np.datetime64("now"))

        return True

    def get_total_samples(self):
        """获取总样本数"""
        if "metadata" in self.file:
            return self.file["metadata"].attrs.get("total_samples", 0)
        return len(self.file["samples"]) if "samples" in self.file else 0

    def list_samples(self):
        """列出所有样本ID"""
        if "samples" in self.file:
            return list(self.file["samples"].keys())
        return []

    def close(self):
        """关闭HDF5文件"""
        if self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def calculate_entropy(logits):
    """计算概率分布的熵"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item()


def process_agentnet(model_path, eval_path, dst, checkpoint_file="outputs.checkpoint"):
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
        HDF5OutputSaver(dst, "a") as saver,
    ):
        # 跳过已经处理的行
        for _ in range(start_line):
            next(f)

        for line_num, line in enumerate(f, start_line + 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            item = json.loads(line)
            id = item["id"]
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

            # 生成文本并获取每个步骤的logits
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    return_dict_in_generate=True,
                    output_scores=True,  # 关键：返回每个生成步骤的分数
                    output_attentions=False,
                    output_hidden_states=False,
                )
            # output_test = processor.decode(outputs[0][inputs["input_ids"].shape[-1] :])
            output_test = processor.decode(
                outputs.sequences[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            # messages.append(
            #     {
            #         "role": "assistant",
            #         "content": [{"type": "text", "text": output_test}],
            #     }
            # )
            # result = {"messages": messages, "teacher_action": item["teacher_action"]}
            # json_line = json.dumps(result, ensure_ascii=False)

            print(output_test)
            saver.save_sample(id, inputs, outputs)
            # 更新检查点（每处理一行就更新）
            with open(checkpoint_file, "w") as f_check:
                f_check.write(str(line_num))
        # 处理完成后删除检查点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        print("处理完成！")


def entropy_test():
    model_path = (
        "/newdata/zhouxy/model/trained_models/uitars_lora_sft_agentnet/merged0911"
    )

    # 初始化模型和processor
    processor = AutoProcessor.from_pretrained(model_path)
    device = torch.device("cuda:3")
    model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action and confidence to complete the task. \n\n## Output Format\n```\nThought: ...\nAction: ...\nConfidence: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1, y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1, y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1, y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n\n## Note\n- Use Chinese in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part. Evaluate your action to be scored, giving it a score from 1 to 5 in 'Confidence' part. A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.\n\n## User Instruction\nCan you help me log into Twitter, search for Elon Musk's profile, and unfollow him if I'm currently following him?",
                },
                {
                    "type": "image",
                    "url": "/newdata/zhouxy/dataset/AgentNet/social_media/images/d0b69adf-b7aa-4c2c-832b-f6a658c7ce68.png",
                },
            ],
        },
    ]

    # 准备输入
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # 生成文本并获取每个步骤的logits
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            return_dict_in_generate=True,
            output_scores=True,  # 关键：返回每个生成步骤的分数
            output_attentions=False,
            output_hidden_states=False,
        )

    # 解码生成的token并计算熵
    generated_token_ids = outputs.sequences[0][inputs["input_ids"].shape[-1] :]
    scores = outputs.scores  # 每个生成步骤的logits

    print("生成过程分析:")
    print("=" * 50)

    for i, (token_id, step_scores) in enumerate(zip(generated_token_ids, scores)):
        token = processor.decode(token_id, skip_special_tokens=True)
        entropy = calculate_entropy(step_scores[0])  # 计算当前步骤的熵

        print(f"步骤 {i + 1}: Token: '{token}' | 熵: {entropy:.4f}")

    # 完整输出
    output_text = processor.decode(
        outputs.sequences[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    print(f"\n完整生成文本: {output_text}")


if __name__ == "__main__":
    model_path = (
        "/newdata/zhouxy/model/trained_models/uitars_lora_sft_agentnet/merged0911"
    )
    eval_path = "/newdata/zhouxy/dataset/AgentNet/social_media/social_media_eval.jsonl"
    dst = "/newdata/zhouxy/model/trained_models/uitars_lora_sft_agentnet/merged0911/process0911.h5"

    process_agentnet(model_path, eval_path, dst)
