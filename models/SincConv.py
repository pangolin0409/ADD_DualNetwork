
# 5. SincConv 模塊
class SincConv(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate):
        super(SincConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sample_rate = sample_rate

        # 允許 f1 和 f2 在訓練時自由學習
        self.f1 = nn.Parameter(torch.linspace(30, sample_rate//2 - 50, out_channels))
        self.f2 = nn.Parameter(torch.linspace(50, sample_rate//2, out_channels))

        # 時間軸，避免 NaN
        self.t = torch.arange(-(self.kernel_size // 2), (self.kernel_size // 2) + 1).float()
        self.t = self.t / (self.sample_rate + 1e-6)  # 避免除以 0
        self.t = self.t.unsqueeze(0).repeat(self.out_channels, 1)

    def sinc(self, x):
        eps = 1e-6  # 避免 `0/0`
        return torch.where(torch.abs(x) < eps, torch.ones_like(x), torch.sin(x) / (x + eps))

    def forward(self, x):
        self.t = self.t.to(x.device)

        # **在 forward() 限制 f1 和 f2，而不是 __init__()**
        self.f1.data = torch.clamp(self.f1, min=30, max=self.sample_rate//2 - 50)
        self.f2.data = torch.clamp(self.f2, min=50, max=self.sample_rate//2)

        filters = []
        for i in range(self.out_channels):
            # 計算 Sinc 濾波器
            low = 2 * self.f1[i] * self.sinc(2 * np.pi * self.f1[i] * self.t[i])
            high = 2 * self.f2[i] * self.sinc(2 * np.pi * self.f2[i] * self.t[i])
            h = high - low

            # 窗函數
            window = 0.54 - 0.46 * torch.cos(2 * np.pi * torch.arange(self.kernel_size) / (self.kernel_size - 1))
            h = h * window.to(x.device)

            filters.append(h.unsqueeze(0).unsqueeze(0))

        filters = torch.cat(filters, dim=0).to(x.device)
        return F.conv1d(x, filters, padding=self.kernel_size // 2)