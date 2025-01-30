import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

class HiddenStateLLMClassifier(nn.Module):
    def __init__(self, llm_name="gpt2", prompt="Classify the audio as Real or Fake", model_dir="./pretrained_models/gpt2_local"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir=model_dir)
        self.llm = AutoModel.from_pretrained(llm_name, cache_dir=model_dir)
        self.prompt_text = prompt

        # 凍結 LLM 參數 (若需要微調則可移除)
        for param in self.llm.parameters():
            param.requires_grad = False

        # 新增二分類線性頭
        self.classifier = nn.Linear(self.llm.config.hidden_size, 2)

    def forward(self, embeddings, task_prompt=None):
        """
        Args:
          embeddings: [batch_size, seq_len, hidden_dim] 的浮點張量 (專家輸出)
          task_prompt: (選擇性) 用來取代預設 prompt_text
        Returns:
          logits: [batch_size, 2]
        """

        # 1) 使用自訂的 prompt 或 fallback 到 self.prompt_text
        if task_prompt is None:
            task_prompt = self.prompt_text

        # 2) 對 prompt 做 tokenizer，取得 prompt_ids
        #    shape: [1, prompt_len]
        prompt_ids = self.tokenizer(task_prompt, return_tensors="pt").input_ids.to(embeddings.device)
        
        # 3) 用 LLM 的 embedding 層把 token ids 轉成 float embedding
        #    shape: [1, prompt_len, hidden_dim]
        prompt_embedding = self.llm.get_input_embeddings()(prompt_ids)

        # 4) 對 batch 展開，使 shape=[batch_size, prompt_len, hidden_dim]
        batch_size = embeddings.size(0)
        prompt_embedding = prompt_embedding.expand(batch_size, -1, -1)

        # 5) 假設專家輸出 embeddings 形狀是 [batch_size, seq_len, hidden_dim]
        #    直接拼接 => shape: [batch_size, (prompt_len + seq_len), hidden_dim]
        combined_input = torch.cat((prompt_embedding, embeddings), dim=1)

        # 6) 使用 inputs_embeds=... 丟給 GPT-2 => 跳過 embedding lookup
        #    outputs.last_hidden_state.shape: [batch_size, (prompt_len + seq_len), hidden_dim]
        outputs = self.llm(inputs_embeds=combined_input)

        # 7) 取最後 token (或其他位置) 來做最終線性分類
        #    這邊選擇最後一個 token => [:, -1, :]
        #    若 seq_len=1 => -1 位置即專家 token
        #    若 prompt_len+seq_len>1 => -1 位置則看你需求
        final_hidden = outputs.last_hidden_state[:, -1, :]  # [batch_size, hidden_dim]

        # 8) 通過線性層得到 [batch_size, 2]
        logits = self.classifier(final_hidden)
        return logits
    
class CustomTokenLLMClassifier(nn.Module):
    def __init__(self, 
                 llm_name="gpt2", 
                 base_prompt="Classify the audio content: {audio_content}", 
                 labels=None, 
                 model_dir="./pretrained_models/gpt2_local"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir=model_dir)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, cache_dir=model_dir)
        
        # 如果沒給 labels，就預設 ["Real", "Fake"]
        if labels is None:
            labels = ["Real", "Fake"]
        self.labels = labels

        # 動態生成 Prompt 和自定義 tokens
        self.base_prompt = base_prompt
        self.custom_tokens = [f"<{label}_prob>" for label in labels]
        # 新增自訂義 token 至 tokenizer，讓 GPT-2 知道它們的 embedding
        self.tokenizer.add_tokens(self.custom_tokens, special_tokens=True)
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # 設置 pad_token 避免生成時出現錯誤
        if self.tokenizer.pad_token is None:
            # 若原本 tokenizer 沒有 pad_token，就以 eos_token 或自定義 token 作為 pad_token
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 凍結 LLM 所有參數（如果只想測試 logits 或做推論）
        for _, param in self.llm.named_parameters():
            param.requires_grad = False

    def forward(self, embeddings: torch.Tensor, task_prompt: str = None):
        """
        Args:
            embeddings (Tensor): [batch_size, audio_seq_len, hidden_dim]
                - 須確保 hidden_dim 與 GPT-2 的 hidden size 一致
            task_prompt (str): 自定義的文字 Prompt，若為 None，則使用預設 base_prompt
        Returns:
            probabilities (Tensor): [batch_size, num_labels]
                - 分別代表 self.labels 中各標籤的概率。
        """

        device = embeddings.device
        
        # 若使用者沒有自定義 Prompt，則用 base_prompt
        if task_prompt is None:
            task_prompt = self.base_prompt.format(audio_content="")

        # 先把 Prompt 轉成 token_ids，再轉成其對應的 embedding
        prompt_ids = self.tokenizer(task_prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)  # [batch_size, prompt_seq_len, hidden_dim]

        # 確認 batch_size 一致
        if prompt_embeds.size(0) != embeddings.size(0):
            raise ValueError(f"Batch size mismatch: prompt_embeds({prompt_embeds.size(0)}) vs embeddings({embeddings.size(0)})")

        # 將 Prompt embedding 和 音訊 embedding 串接
        # combined_embeds = torch.cat([prompt_embeds, embeddings], dim=1)  # [batch_size, prompt_seq_len+audio_seq_len, hidden_dim]

        # attention_mask 要跟 combined_embeds 尺寸對應
        # attention_mask = torch.ones(combined_embeds.size()[:-1], dtype=torch.long, device=device)  # [batch_size, seq_len]

        # 透過 generate，讓模型「再生成」幾個 token（數量 = len(self.custom_tokens)）
        outputs = self.llm.generate(
            input_ids=prompt_ids,
            max_new_tokens=len(self.custom_tokens),
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        # 拿到每一步生成的 logits
        scores_list = outputs.scores  # len = max_new_tokens；每個元素 shape = [batch_size, vocab_size]
        # 拿到最終生成的序列
        generated_ids = outputs.sequences[0]  # tensor of shape (prompt_length + 2,)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        print("Generated sequences:", generated_text)
        # 根據自定義 token（<Real_prob>, <Fake_prob>...），把對應 logits 抽取出來
        logits = []
        for i, token in enumerate(self.custom_tokens):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            # scores_list[i] 是第 i 步產生的 logits => [batch_size, vocab_size]
            # 我們只關心 token_id 對應的 logit
            token_logit = scores_list[i][:, token_id].unsqueeze(-1)  # [batch_size, 1]
            logits.append(token_logit)

        # 堆疊成 [batch_size, num_labels]
        logits = torch.cat(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)

        return probabilities
    
class GPT2RegressionHead(nn.Module):
    def __init__(self, llm_name="gpt2", 
                  model_dir="./pretrained_models/gpt2_local",  extra_token="<prob>"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir=model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(llm_name, cache_dir=model_dir)
        
        self.extra_token = extra_token
        # 找到該 token 對應的 ID
        self.extra_token_id = self.tokenizer.convert_tokens_to_ids(self.extra_token)

    def forward(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        # 在結尾加上 <prob>
        prob_id = self.tokenizer.convert_tokens_to_ids("<prob>")
        input_ids = torch.cat([input_ids, torch.tensor([[prob_id]])], dim=-1)

        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True
        )

        # output.scores[0] 會是第一個新token的 logits，shape = [batch_size, vocab_size]
        # 我們只看 extra_token 的那一維
        regression_logit = output.scores[0][0, prob_id]  # 假設 batch_size=1
        prob = torch.sigmoid(regression_logit)
        return prob