import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import tiktoken

enc = tiktoken.get_encoding("r50k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

LR = 1e-3
EMBEDDING_VOCAB = enc.n_vocab
MAX_SEQ_LEN=64
ACCUMULATION_STEPS = 3
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device
# Model CONFIG
MODEL_EMBED_DIM = 750
MODEL_NUM_HEADS = 10
MODEL_LATENT_DIM = 362
MODEL_TRANSFORMER_BLOCKS = 64

class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        residual = self.norm1(x)
        residual, _ = self.attn(residual, residual, residual, attn_mask = mask)
        x = x + self.dropout(residual)
        residual = self.norm2(x)
        residual = self.mlp(residual)
        x = x + self.dropout(residual)
        return x

# %%
class CrossAttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


# %%
class SentenceVAE(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, latent_dim: int, transformer_blocks: int, dropout = 0.1):
        super().__init__()
        # encoder
        self.embed = nn.Embedding(EMBEDDING_VOCAB, embedding_dim)
        self.pos_embed = nn.Embedding(MAX_SEQ_LEN, embedding_dim)
        self.attn_layers: nn.ModuleList = nn.ModuleList([SelfAttentionLayer(embedding_dim, num_heads) for _ in range(transformer_blocks)])
        # self.attn1 = SelfAttentionLayer(embedding_dim, num_heads)
        # self.attn2 = SelfAttentionLayer(embedding_dim, num_heads)
        # self.attn3 = SelfAttentionLayer(embedding_dim, num_heads)
        self.fc_mean = nn.Linear(embedding_dim, latent_dim)
        self.fc_log_var = nn.Linear(embedding_dim, latent_dim)

        # decoder
        self.fc_in = nn.Linear(latent_dim, embedding_dim)
        self.crs_attn_layers: nn.ModuleList = nn.ModuleList([CrossAttentionLayer(embedding_dim, num_heads) for _ in range(transformer_blocks)])
        # self.crs_attn1 = CrossAttentionLayer(embedding_dim, num_heads)
        # self.crs_attn2 = CrossAttentionLayer(embedding_dim, num_heads)
        # self.crs_attn3 = CrossAttentionLayer(embedding_dim, num_heads)
        self.norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, EMBEDDING_VOCAB, bias=False)

        self.lm_head.weight = self.embed.weight


    # Returns (1, embed_dim)
    def encode(self, x: torch.Tensor):
        positions = torch.arange(x.size(0), device=x.device).unsqueeze(1)
        x = self.embed(x) + self.pos_embed(positions)
        for block in self.attn_layers:
            x = block(x)
        pooled = x.mean(dim = 1)
        mu = self.fc_mean(pooled)
        log_var = self.fc_log_var(pooled)
        return x, mu, log_var

    def decode(self, z: torch.Tensor, seq_len) -> torch.Tensor:
        x = self.fc_in(z).unsqueeze(1).expand(-1, seq_len, -1).clone()
        for block in self.crs_attn_layers:
            x = block(x)
        # x = self.crs_attn1(x)
        # x = self.crs_attn2(x)
        # x = self.crs_attn3(x)
        return self.lm_head(self.norm(x))  # (batch, seq, vocab_size)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + torch.rand_like(std) * std
        return mu

    def forward(self, x):
        encoder_out, mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        logits = self.decode(z, encoder_out.size(1))

        recon_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())

        return logits, recon_loss, kl_loss

# %%
class SentenceDataset(Dataset):
    def __init__(self, filepath, max_len=MAX_SEQ_LEN, max_points=500_000):
        self.enc = enc
        self.max_len = max_len
        self.filepath = filepath
        self.max_points = max_points
        self.offsets = self._build_offsets()


    def _build_offsets(self):
        offsets = []
        with open(self.filepath, "rb") as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)
                if len(offsets) > self.max_points:
                    break
        return offsets

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.filepath, "r") as f:
            f.seek(self.offsets[idx])
            sentence = f.readline().strip()
        tokens = self.enc.encode(sentence)[:self.max_len]
        tokens += [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

scaler = GradScaler("cuda")

def train(model, dataloader, optimizer, device, beta=1.0):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0

    pbar = tqdm(dataloader, desc="training")
    optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        batch = batch.to(device)

        with autocast("cuda"):
            logits, recon_loss, kl_loss = model(batch)
            loss = (recon_loss + 0.001 * kl_loss) / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUMULATION_STEPS
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        pbar.set_postfix({
            "loss": f"{total_loss / (i + 1):.4f}",
            "recon": f"{total_recon / (i + 1):.4f}",
            "kl": f"{total_kl / (i + 1):.4f}",
        })

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_kl / n

# --- KL annealing ---
def get_beta(epoch, warmup_epochs=40):
    # gradually increase KL weight so model doesn't collapse early
    return min(1.0, epoch / warmup_epochs)


# --- Put it all together ---

datasetcls = SentenceDataset("fineweb.jsonl", max_points=100_000)

train_size = int(0.9 * len(datasetcls))
val_size = len(datasetcls) - train_size
train_dataset, val_dataset = random_split(datasetcls, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

model = SentenceVAE(
    embedding_dim=MODEL_EMBED_DIM,
    num_heads=MODEL_NUM_HEADS,
    latent_dim=MODEL_LATENT_DIM,
    transformer_blocks=MODEL_TRANSFORMER_BLOCKS,
).to(device)
checkpoint_name = f"checkpoint_{MODEL_EMBED_DIM}{MODEL_LATENT_DIM}{MODEL_TRANSFORMER_BLOCKS}.pt"
if os.path.isfile(checkpoint_name):
    model.load_state_dict(torch.load(checkpoint_name, map_location=DEVICE))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

try:
    for epoch in range(50):
        beta = get_beta(epoch, warmup_epochs=10)
        train_loss, recon, kl = train(model, train_loader, optimizer, device, beta)
        scheduler.step()
        print(f"epoch {epoch+1} | train {train_loss:.4f} | recon {recon:.4f} | kl {kl:.4f} | beta {beta:.2f}")
        torch.save(model.state_dict(), checkpoint_name)
except KeyboardInterrupt:
    pass

torch.save(model.state_dict(), checkpoint_name)

def to_words(logits):
    token_ids = logits.argmax(dim=-1)
    token_ids = token_ids[0].tolist()
    print(token_ids)
    token_ids = [t for t in token_ids if t != 0]
    print(enc.decode(token_ids))

model.eval()

while (req := input("RQ: ")) != "q":
    test_sentence = torch.LongTensor(enc.encode(req)).reshape(1, -1).to(DEVICE)
    logits, _, _ = model(test_sentence)
    to_words(logits)

# %%

