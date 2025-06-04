import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 加载数据集
dataset = load_dataset('imdb')

# 加载预训练的 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 对数据集进行分词
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 设置格式化数据集
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 加载预训练的 BERT 模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

# 开始训练
trainer.train()
