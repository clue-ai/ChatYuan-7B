
**ChatYuan-7B**是一个支持中英双语的功能型对话语言大模型。它是基于LLaMA-7B模型上继续进行三阶段训练的模型。
三阶段如下：
1. 在中文通用语料上继续预训练500亿中文token
2. 在数百种任务集上进行任务式指令微调训练
3. 在人类反馈数据集上进行指令微调训练


## 更多细节参考[GitHub](https://github.com/clue-ai/ChatYuan-7B)

## 使用方式
为了遵守LLaMA模型许可证，我们将ChatYuan-7B权重发布为增量权重。您可以将我们的增量权重与原始的LLaMA权重相加，得到ChatYuan-7B权重。

1. 通过原始[LLaMA-7B](https://github.com/facebookresearch/llama)生成LLaMA的hf模型(LLaMA-7B-HF)，可以参考[指导](https://huggingface.co/docs/transformers/main/model_doc/llama)
2. 合并LLaMA-7B的hf模型和ChatYuan-7B模型
### 合并脚本
```shell
python3 apply_delta.py --base ~/model_weights/LLaMA-7B-HF --delta ~/model_weights/ChatYuan-7B --target ~/model_weights/ChatYuan-7B-merge
```

## 加载方式

```python
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import sys
ckpt = "~/model_weights/ChatYuan-7B-merge"
device = torch.device('cuda')
model = LlamaForCausalLM.from_pretrained(ckpt)
tokenizer = AutoTokenizer.from_pretrained(ckpt)
```

## 推理方式

```python
prompt = "用户:  \n小元: "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
generate_ids = model.generate(input_ids, max_new_tokens=1024, do_sample = True, temperature = 0.7)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
response = output[len(prompt):]
print(response)
```


## 限制

在当前基础模型和数据训练的模型中仍存在一些问题：

1. 当要求遵循与事实相关的指令时，模型可能会生成事实错误。
  
2. 由于模型仍然难以识别潜在的有害指令，偶尔会生成有害的回应。
  
3. 在推理和编码方面仍然需要改进

由于模型仍然存在限制，我们要求开发者只能将开源代码、数据、模型以及通过该项目生成的其他任何成果用于研究目的。不允许商业用途和其他潜在有害的使用场景。
