```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from typing import Optional, Tuple
import numpy as np

# 配置类（简化版 GPT-2 small）
class GPT2Config:
    def __init__(
        self,
        vocab_size=50257,  # GPT-2 词表大小
        n_positions=1024,  # 最大序列长度
        n_embd=768,        # 嵌入维度
        n_layer=12,        # Transformer层数
        n_head=12,         # 注意力头数
        n_inner=None,      # FFN中间维度
        activation_function="gelu",
        resid_pdrop=0.1,   # 残差dropout
        embd_pdrop=0.1,    # 嵌入层dropout
        attn_pdrop=0.1,    # 注意力dropout
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = 4 * n_embd if n_inner is None else n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


# 因果自注意力掩码
def causal_attention_mask(seq_len, device):
    """生成下三角因果掩码"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask = mask.view(1, 1, seq_len, seq_len)
    return mask


# 注意力机制
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Q, K, V 投影
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # 注册缓冲区用于缓存
        self.register_buffer("bias", causal_attention_mask(config.n_positions, "cpu"))

    def forward(self, x, attention_mask=None, use_cache=False, past_key_value=None):
        batch_size, seq_len, _ = x.shape
        
        # 生成 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 重塑为多头
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # 如果有过去的键值对，连接起来
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        
        # 缓存当前的键值对
        if use_cache:
            present = (k, v)
        else:
            present = None
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用因果掩码
        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))
        
        # 应用额外的注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, v)
        
        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_embd)
        
        # 输出投影
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, present


# 前馈网络
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# Transformer 块
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None, use_cache=False, past_key_value=None):
        # 自注意力层
        attn_output, present = self.attn(
            self.ln_1(x),
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        x = x + attn_output
        
        # 前馈网络层
        mlp_output = self.mlp(self.ln_2(x))
        x = x + mlp_output
        
        return x, present


# GPT-2 模型
class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 词嵌入和位置嵌入
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer 层
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # 最终层归一化
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 位置 ids
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 输入嵌入
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        # 准备注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # 初始化过去的键值对
        if past_key_values is None:
            past_key_values = [None] * len(self.h)
        
        presents = [] if use_cache else None
        
        # 通过所有 Transformer 层
        for i, (block, past_key_value) in enumerate(zip(self.h, past_key_values)):
            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value
            )
            if use_cache:
                presents.append(present)
        
        # 最终层归一化
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states, presents


# GPT-2 语言模型（带 LM Head）
class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重绑定
        self.lm_head.weight = self.transformer.wte.weight
        
        # 初始化 LM head
        self.lm_head.weight.data.normal_(mean=0.0, std=config.initializer_range)
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=False
    ):
        # 获取隐藏状态
        hidden_states, presents = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        # LM head
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # 移位以便预测下一个词
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            "loss": loss,
            "logits": lm_logits,
            "past_key_values": presents
        }
    
    def generate(
        self,
        input_ids,
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.0,
        num_return_sequences=1
    ):
        """生成文本"""
        self.eval()
        generated = input_ids
        
        with torch.no_grad():
            past_key_values = None
            
            for _ in range(max_length - input_ids.shape[1]):
                # 前向传播
                outputs = self(
                    generated,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs["logits"][:, -1, :] / temperature
                past_key_values = outputs["past_key_values"]
                
                # 重复惩罚
                if repetition_penalty != 1.0:
                    for token_id in set(generated[0].tolist()):
                        logits[:, token_id] /= repetition_penalty
                
                # Top-k 过滤
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                
                # Top-p (nucleus) 过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过 top_p 的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float('Inf')
                
                # 采样或贪心选择
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # 如果生成结束符，停止生成
                if next_token.item() == self.config.eos_token_id:
                    break
        
        return generated


# 2. 训练代码
class GPT2Trainer:
    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-5,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000
        )
        
    def train_step(self, batch):
        """单个训练步骤"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        labels = input_ids.clone()
        
        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs["loss"]
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# 3. 数据加载和预处理
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 简单的字符级tokenizer（简化版）
        tokens = [ord(c) % 50000 for c in text[:self.max_length]]
        
        # 填充到固定长度
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor([1] * min(len(tokens), self.max_length) + 
                                          [0] * max(0, self.max_length - len(tokens)), 
                                          dtype=torch.long)
        }


# 4. 示例：创建和训练模型
def create_and_train_gpt2():
    # 创建配置
    config = GPT2Config(
        vocab_size=50000,  # 简化词表大小
        n_positions=512,   # 简化序列长度
        n_embd=256,        # 简化嵌入维度
        n_layer=6,         # 简化层数
        n_head=8           # 简化头数
    )
    
    # 创建模型
    model = GPT2LMHeadModel(config)
    
    # 创建训练器
    trainer = GPT2Trainer(model, config)
    
    # 示例数据
    texts = [
        "Hello, how are you?",
        "I am learning about GPT-2.",
        "This is a simple implementation.",
        "Transformers are powerful models.",
        "Machine learning is fascinating."
    ]
    
    # 创建数据集
    dataset = TextDataset(texts, tokenizer=None, max_length=config.n_positions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            loss = trainer.train_step(batch)
            total_loss += loss
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # 保存模型
    trainer.save_model("gpt2_model.pth")
    
    return model, trainer


# 5. 生成文本示例
def generate_text_example():
    # 创建配置（与训练时相同）
    config = GPT2Config(
        vocab_size=50000,
        n_positions=512,
        n_embd=256,
        n_layer=6,
        n_head=8
    )
    
    # 创建模型
    model = GPT2LMHeadModel(config)
    
    # 加载预训练权重（如果有）
    # model.load_state_dict(torch.load("gpt2_model.pth"))
    
    # 生成文本
    input_text = "The future of AI is"
    
    # 简单的tokenization
    input_ids = torch.tensor([[ord(c) % 50000 for c in input_text]], dtype=torch.long)
    
    # 生成
    generated_ids = model.generate(
        input_ids,
        max_length=50,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    
    # 解码
    generated_text = ''.join([chr(token % 50000) for token in generated_ids[0].tolist()])
    
    print("Input:", input_text)
    print("Generated:", generated_text)
    
    return generated_text


# 6. 高级功能：注意力可视化
def visualize_attention(model, input_text, layer_idx=0, head_idx=0):
    """可视化特定层的注意力权重"""
    model.eval()
    
    # Tokenize
    input_ids = torch.tensor([[ord(c) % 50000 for c in input_text]], dtype=torch.long)
    
    # 获取注意力权重（需要修改Attention类来返回权重）
    with torch.no_grad():
        outputs = model.transformer(input_ids, use_cache=False)
    
    print(f"Input text: {input_text}")
    print(f"Length: {len(input_text)}")
    
    # 注意：这里简化了，实际需要修改模型来返回注意力权重
    return outputs


# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("GPT-2 从零实现")
    print("=" * 60)
    
    # 1. 创建和训练模型
    print("\n1. 创建和训练模型...")
    model, trainer = create_and_train_gpt2()
    
    # 2. 生成文本示例
    print("\n2. 生成文本示例...")
    generated_text = generate_text_example()
    
    # 3. 模型信息
    print("\n3. 模型信息:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 4. 测试推理速度
    print("\n4. 测试推理速度...")
    import time
    
    test_input = torch.randint(0, 50000, (1, 10))
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(test_input)
    end_time = time.time()
    
    print(f"推理时间: {(end_time - start_time)*1000:.2f}ms")
    
    print("\n" + "=" * 60)
    print("GPT-2 实现完成！")
    print("=" * 60)
```

