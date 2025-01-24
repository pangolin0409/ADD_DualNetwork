import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class PromptedLLMClassifier(nn.Module):
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
