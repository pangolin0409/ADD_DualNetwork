import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from models.rawnet3.RawNet3 import RawNet3
from models.rawnet3.RawNetBasicBlock import Bottle2neck
from models.classifier.AudioClassifier import SimpleMLPClassifier

class HierMoEModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        device = args.device

        # 1) RawNet3
        self.rawnet = RawNet3(
            Bottle2neck,
            model_scale=8,
            context=True,
            summed=True,
            encoder_type="ECA",
            nOut=256,
            out_bn=False,
            sinc_stride=10,
            log_sinc=True,
            norm_sinc="mean",
            grad_mult=1.0,
        )
        # load
        raw_ckpt = torch.load(args.rawnet3_path, map_location=device)
        self.rawnet.load_state_dict(raw_ckpt["model"])
        self.rawnet.requires_grad_(False)

        # 2) wav2vec
        self.wav2vec = Wav2Vec2Model.from_pretrained(args.wav2vec_path)
        for p in self.wav2vec.parameters():
            p.requires_grad = False

        # local MoE for raw
        self.localMoE_raw = LocalMoE(input_dim=256, hidden_dim=256, n_experts=2)
        # local MoE for wav2vec
        hidden_size = self.wav2vec.config.hidden_size
        self.localMoE_wav = LocalMoE(input_dim=hidden_size, hidden_dim=256, n_experts=2)

        # global MoE
        # => concat raw(256) + w2v(256) => 512 => gating => 256
        self.globalMoE = GlobalMoE(input_dim=512, output_dim=256, n_experts=2)

        # final classifier
        self.final_classifier = SimpleMLPClassifier(
            input_dim=256,
            hidden_dim=128,
            num_classes=2
        )

    def forward_raw_only(self, wave):
        """只跑 rawNet + localMoE, 用於 Phase1 單獨訓練 raw local"""
        # rawnet => [B,256]
        if wave.dim()==2: # [B,T]
            wave = wave.unsqueeze(1) # => [B,1,T]
        raw_emb = self.rawnet(wave)  # [B,256]
        # local MoE => out
        out, _ = self.localMoE_raw(raw_emb)
        return out

    def forward_wav_only(self, wave):
        """只跑 wav2vec + localMoE, 用於 Phase1 單獨訓練 w2v local"""
        if wave.dim()==3:
            wave = wave.squeeze(1)  # [B,T]
        w2v_out = self.wav2vec(wave)
        # 取 last_hidden => [B,seq,hidden_size]
        hidden = w2v_out.last_hidden_state
        x = hidden.mean(dim=1)  # [B, hidden_size]
        out, _ = self.localMoE_wav(x)
        return out

    def forward_global(self, raw_out, w2v_out):
        """融合 raw_out & w2v_out => globalMoE => final classifier"""
        fused = torch.cat([raw_out, w2v_out], dim=-1) # => [B,512]
        glo_out, _ = self.globalMoE(fused)
        logits = self.final_classifier(glo_out)
        return logits

    def forward_all(self, wave):
        """
        完整路徑: raw => local, w2v => local => global => classifier
        """
        # raw
        if wave.dim()==2:
            wave_r = wave.unsqueeze(1)
        else:
            wave_r = wave
        raw_emb = self.rawnet(wave_r)
        raw_out, raw_info = self.localMoE_raw(raw_emb)

        # w2v
        if wave.dim()==3:
            wave_w = wave.squeeze(1)
        else:
            wave_w = wave
        w2v_outs = self.wav2vec(wave_w)
        hidden = w2v_outs.last_hidden_state
        x = hidden.mean(dim=1)
        w2v_out, w2v_info = self.localMoE_wav(x)

        # global
        fused = torch.cat([raw_out, w2v_out], dim=-1)
        glo_out, glo_info = self.globalMoE(fused)
        logits = self.final_classifier(glo_out)
        return logits, (raw_info, w2v_info, glo_info)


# -------------------------------------------
# local / global MoE modules
# -------------------------------------------
class LocalMoE(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, n_experts=2):
        super().__init__()
        self.gate_fc = nn.Linear(input_dim, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        gate_logits = self.gate_fc(x)
        gate_scores = F.softmax(gate_logits, dim=-1)
        out = torch.zeros_like(x)
        for i, exp in enumerate(self.experts):
            w = gate_scores[:, i].unsqueeze(-1)
            e_out = exp(x)
            out += w*e_out
        return out, (gate_scores, gate_logits)

class GlobalMoE(nn.Module):
    def __init__(self, input_dim=512, output_dim=256, n_experts=2):
        super().__init__()
        self.gate_fc = nn.Linear(input_dim, n_experts)
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(n_experts)
        ])

    def forward(self, x):
        gate_logits = self.gate_fc(x)
        gate_scores = F.softmax(gate_logits, dim=-1)
        out = torch.zeros(x.size(0), self.experts[0].out_features, device=x.device)
        for i,exp in enumerate(self.experts):
            w = gate_scores[:,i].unsqueeze(-1)
            e_out = exp(x)
            out += w*e_out
        return out, (gate_scores, gate_logits)
