import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, HubertModel, Wav2Vec2FeatureExtractor
from sklearn.cluster import KMeans
# ========================
# Feature Extractors
# ========================
class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-large-960h", output_dim=256):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        
        self.linear = nn.Linear(self.model.config.hidden_size, output_dim)  # 1024 -> 256

    def forward(self, x):
        with torch.no_grad():  # Frozen SSL Model
            input_values = self.wav2vec_extractor(
                x, 
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values  # (batch, time_steps)
            input_values = input_values.squeeze(0)
            features = self.model(input_values).last_hidden_state  # (batch, time_steps, 1024)
        return self.linear(features)  # (batch, time_steps, 256)

class HubertFeatureExtractor(nn.Module):
    def __init__(self, model_name="facebook/hubert-large-ls960-ft", output_dim=128, num_clusters=100):
        super().__init__()
        self.model = HubertModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.model.config.hidden_size, output_dim)  # 1024 -> 128
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        
    def forward(self, x):
        with torch.no_grad():  # Frozen SSL Model
            features = self.model(x).last_hidden_state  # (batch, time_steps, 1024)
        embeddings = self.linear(features)  # (batch, time_steps, 128)
        # Apply k-means clustering to discretize features
        cluster_ids = self.kmeans.fit_predict(embeddings.view(-1, embeddings.shape[-1]))
        cluster_embeddings = torch.tensor(cluster_ids, dtype=torch.float32).view(embeddings.shape[:-1])  # Reshape back
        return cluster_embeddings