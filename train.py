import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def compute_metrics(eval_pred: tuple):
    # 将 eval_pred 的元素转换为 tensor
    logits = torch.tensor(eval_pred[0])
    labels = torch.tensor(eval_pred[1])
    
    # 对 logits 应用 sigmoid 得到概率
    probs = torch.sigmoid(logits)
    # 使用阈值 0.5 转换为二分类预测结果
    preds = (probs > 0.5).float()
    
    # 计算准确率：逐元素比较后求平均
    accuracy = (preds == labels).float().mean().item()
    
    return {"accuracy": accuracy}

def train():
    with open("./datasets.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    with open("./config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        
    pre_trained = config["model"]
    output = config["output"]
    
    # 将数据构造成 HuggingFace datasets 格式
    dataset = Dataset.from_dict(data)
    split_dataset = dataset.train_test_split(test_size=0.1, shuffle=True)
    
    print("Fetching pre trained model...")
    
    # 加载预训练模型
    tokenizer = BertTokenizer.from_pretrained(pre_trained)
    
    # 定义分词函数
    def tokenize_function(examples: object):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    # 对数据进行分词
    tokenized_datasets = split_dataset.map(tokenize_function, batched=True)
    
    # 3. 加载预训练模型，并指定输出维度为2，并设置问题类型为多标签分类
    model = BertForSequenceClassification.from_pretrained(
        pre_trained,
        num_labels=2,  # 两个二分类标签
        problem_type="multi_label_classification"  # 使用 BCEWithLogitsLoss 进行多标签训练
    )
    
    # 4. 定义训练参数
    training_args = TrainingArguments(
        output_dir=output,
        evaluation_strategy="epoch",
        num_train_epochs=config["epoches"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=len(data["text"]) // 10,
        save_steps=config["epoches"] * len(data["text"])
    )
    
    # 5. 使用 Trainer 进行微调
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics
    )
    
    print("Start to train...")
    
    # 开始训练
    trainer.train()

    # 保存模型和分词器
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)

if __name__ == "__main__":
    train()