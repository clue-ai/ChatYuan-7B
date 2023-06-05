
**ChatYuan-7B**是一个支持中英双语的功能型对话语言大模型。它是基于LLaMA-7B模型上继续进行三阶段训练的模型。
三阶段如下：
1. 在中文通用语料上继续预训练500亿中文token
2. 在数百种任务集上进行任务式指令微调训练
3. 在数百万人类反馈数据集上进行指令微调训练


## [在线DEMO.Huggingface](https://huggingface.co/spaces/ClueAI/ChatYuan-7B)

## 使用方式
为了遵守LLaMA模型许可证，我们将ChatYuan-7B权重发布为增量权重。您可以将我们的增量权重与原始的LLaMA权重相加，得到ChatYuan-7B权重。

1. 通过原始[LLaMA-7B](https://github.com/facebookresearch/llama)生成LLaMA的hf模型(LLaMA-7B-HF)，可以参考[指导](https://huggingface.co/docs/transformers/main/model_doc/llama), 也可以直接使用[llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)
2. 合并LLaMA-7B的hf模型和ChatYuan-7B模型
### 合并脚本
```shell
python3 apply_delta.py --base ~/model_weights/llama-7b-hf --delta ~/model_weights/ChatYuan-7B --target ~/model_weights/ChatYuan-7B-merge
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
def answer(prompt):
  prompt = f"用户：{prompt}\n小元："
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
  generate_ids = model.generate(input_ids, max_new_tokens=1024, do_sample = True, temperature = 0.7)
  output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  response = output[len(prompt):]
  return response
```
 
## 示例

```
input: 学前教育专业岗位实习中，在学生方面会存在问题，请提出改进措施。800字
output: 作为学前教育专业的学生，在岗位实习中可能会遇到各种各样的问题，这些问题不仅影响了我们的学习和成长，也影响了我们的教育质量。为了帮助学生更好地完成实习工作，我们提出以下改进措施：

一、提高教育质量

在学前教育专业岗位实习中，教育质量是关键。学生应该掌握教育方法和技能，并且在实践中学习。为了提高教育质量，学校应该从以下几个方面着手：

1. 提供高质量的教育资源。学校应该提供各种教育资源，包括教材、工具和软件等，以帮助学生更好地完成实习工作。

2. 提高教师质量。教师应该具备丰富的教育经验和专业知识，以便更好地指导学生。

3. 提供培训和评估。学校应该为学生提供培训和评估，以帮助学生更好地掌握技能和知识。

二、加强沟通与合作

在岗位实习中，学生应该与老师和同学进行沟通和合作。老师应该与学生保持良好的沟通，以便更好地指导学生。同时，学生也应该与同学进行合作，以便更好地完成实习工作。

为了加强沟通与合作，学校应该从以下几个方面着手：

1. 提供交流渠道。学校应该为学生提供交流渠道，包括线上和线下的交流方式，以便学生与老师和同学之间进行良好的沟通。

2. 提供团队合作机会。学校应该为学生提供团队合作机会，以便学生能够更好地了解和协作。

3. 提供培训和指导。学校应该为学生提供培训和指导，以便学生能够更好地理解和应对问题。

三、保证实习安全

在岗位实习中，学生应该保证自身的安全。学校应该为学生提供安全保障，包括在实习过程中提供帮助和指导，以及在紧急情况下提供支持和帮助。

为了保证实习安全，学校应该从以下几个方面着手：

1. 提供安全教育。学校应该为学生提供安全教育，以便学生能够更好地了解安全要求和安全方法。

2. 提供安全指导。学校应该为学生提供安全指导，以便学生能够更好地掌握实习安全方法。

3. 提供安全保障。学校应该为学生提供安全保障，包括在实习过期中提供帮助和指导，以及在紧急情况下提予支持和帮助。

总之，学前教育专业岗位实习是学生成长和发展的重要阶段。通过提高教育质量、加强沟通与合作和保证实习安全等措施，学校可以帮助学生更好地完成实习工作，提高教育质量和专业技能。
```

```
input: 根据标题生成文章：标题：屈臣氏里的化妆品到底怎么样？正文：化妆品，要讲究科学运用，合理搭配。屈臣氏起码是正品连锁店。请继续后面的文字。
output: 作为一家有着数十年历史的正品连锁店，屈臣氏在市场上拥有一定的口碑和信誉。而对于喜欢化妆品的女性来说，屈臣氏里的化妆品也是不可忽视的选择。那么，屈臣氏里的化妆品到底怎么样呢？

首先，我们需要了解一下屈臣氏的采购政策。屈臣氏一直秉持着“质量第一、信誉商誉、顾客至上”的经营理念，严格按照国家的相关质量标准进行采购管理。在采购过程中，屈臣氏会对每一家供应商进行详细的评估和审核，以确保所采购的商品符合质量标准。

其次，我们需要了解一下屈臣氏销售的化妆品品牌。屈臣氏销售的化妆品品牌非常丰富，包括国内外著名品牌如兰蔻、雅诗兰黛、娇兰、资生堂等。屈臣氏还会不定期地邀请国内外知名化妆品专家来店内进行化妆品推介和使用指导，为顾客提供更专业的服务。

最后，我们需要了解一下屈臣氏销售的化妆品的价格。相比于一些高端专业的化妆品店，屈臣氏的化妆品价格相对较为亲民，但在保证品质的前提下，仍然可以满足大多数消费者的需求。

总之，屈臣氏里的化妆品在品质、销售渠道和价格等方面都具有一定的优势。但是，我们在选择化妆品时，还是需要注重科学的运用和合理的搭配，以达到更好的化妆效果。

```

```
input: 帮我写一个英文营销方案，针对iphone
output: Marketing Plan for iPhones

Introduction
iPhones have become the flagship devices of Apple and have gained immense popularity worldwide. With a sleek design, advanced features, and a smooth user interface, iPhones attract and retain customers through their lifetime. Therefore, developing a marketing plan for iPhones is essential to ensure the continued success and growth of the brand.

Objectives
The primary objective of this marketing plan is to increase the brand awareness and demand for iPhones by targeting potential customers and creating a strong relationship with them.

Target Market
The target audience for this marketing plan is people who are in the market for a new smartphone. The target audience is classified into three segments:

1. Upgraders: People who are currently using older versions of iPhones and are looking for a new device with advanced features.

2. Switchers: People who are currently using Android smartphones and are looking to switch to iPhones.

3. Newbies: People who have never used an iPhone before and are looking to experience its features for the first time.

Tactics
To achieve the objectives, the following tactics will be implemented:

1. Advertising
Apple's advertising strategy is well-known for its focus on creating an emotional connection with the audience through its advertisements. The marketing campaign will focus on highlighting the advanced features of iPhones and the benefits of using them. The advertisements will be placed on various platforms, including social media, television, and billboards.

2. Product Demonstrations
Product demonstrations will be held at Apple stores and other retail outlets to allow potential customers to experience the advanced features of iPhones. This will help people make informed decisions about their next smartphone.

3. Referral Program
A referral program will be launched to encourage existing customers to refer their friends to the brand. This will help to attract new customers and retain existing ones.

4. Content Marketing
Content marketing will be used to create and share valuable content that will be useful to potential customers. This will include tutorials, guides, and other content that will help people make informed decisions about their next smartphone.

Conclusion
In conclusion, this marketing plan will help Apple achieve its goals of increasing brand awareness and demand for its products. The tactics employed in this plan will help to attract new customers and retain existing ones by creating a strong relationship with the brand.
```


## 限制

在当前基础模型和数据训练的模型中仍存在一些问题：

1. 当要求遵循与事实相关的指令时，模型可能会生成事实错误。
  
2. 由于模型仍然难以识别潜在的有害指令，偶尔会生成有害的回应。
  
3. 在推理和编码方面仍然需要改进

由于模型仍然存在限制，我们要求开发者只能将开源代码、数据、模型以及通过该项目生成的其他任何成果用于研究目的。不允许商业用途和其他潜在有害的使用场景。


## ChatYuan-7B讨论与交流

<p float="left">   
  <img src="https://github.com/clue-ai/ChatYuan-7B/blob/main/resources/img/chatyuan-7b.jpeg"  width="30%" height="30%"></img>
  <img src="https://github.com/clue-ai/ChatYuan-7B/blob/main/resources/img/brightmart.jpeg"  width="30%" height="30%"></img>
</p> 
