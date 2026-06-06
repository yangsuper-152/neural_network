import torch
import torch.nn.functional as F

# ---------- 数据准备(两个模型共用) ----------
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}


# ============================================================
# 模型一:计数模型 (counting model)
# ============================================================
# 直接统计每个 bigram 出现的次数,然后归一化成概率表 P
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

P = (N + 1).float()              # +1 是平滑(smoothing),避免概率为 0
P /= P.sum(1, keepdims=True)     # 每一行归一化成概率分布

print('=== 计数模型生成 ===')
g = torch.Generator().manual_seed(2147483647)
for i in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

# 计算计数模型在整个数据集上的 loss(平均负对数似然)
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        prob = P[stoi[ch1], stoi[ch2]]
        log_likelihood += torch.log(prob)
        n += 1
print(f'计数模型 loss = {-log_likelihood / n:.4f}\n')


# ============================================================
# 模型二:神经网络模型 (neural net)
# ============================================================
# 构造训练集
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples:', num)

# 初始化权重(27x27)
g = torch.Generator().manual_seed(2147483647 + 2)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# 梯度下降
for k in range(200):
    xenc = F.one_hot(xs, num_classes=27).float()       # one-hot 编码输入
    logits = xenc @ W                                  # 预测 log-counts
    counts = logits.exp()                              # 相当于 N
    probs = counts / counts.sum(1, keepdims=True)      # softmax
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W ** 2).mean()
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad
print(f'神经网络 loss = {loss.item():.4f}\n')

print('=== 神经网络生成 ===')
g = torch.Generator().manual_seed(2147483647)
for i in range(10):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))