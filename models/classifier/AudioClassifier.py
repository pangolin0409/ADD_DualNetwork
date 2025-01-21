import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch
class PromptedLLMClassifier(nn.Module):
    def __init__(self, llm_name="gpt2", prompt="Classify the audio as Real or Fake"):
        super(PromptedLLMClassifier, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModel.from_pretrained(llm_name)
        self.prompt = prompt

        # Classification head
        self.classifier = nn.Linear(self.llm.config.hidden_size, 2)

    def forward(self, embeddings, task_prompt=None):
        # Prepare Prompt
        prompt_text = task_prompt if task_prompt else self.prompt
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
        prompt_embedding = self.llm.get_input_embeddings()(prompt_ids)

        # Combine Prompt and Audio Embeddings
        combined_input = torch.cat((prompt_embedding, embeddings), dim=1)

        # Forward Pass through LLM
        outputs = self.llm(inputs_embeds=combined_input)
        logits = self.classifier(outputs.last_hidden_state[:, -1, :])
        return logits
