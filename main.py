import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset


    return DataLoader, Dataset, F, nn, torch


@app.cell
def _():
    import tiktoken
    enc = tiktoken.get_encoding("r50k_base")
    assert enc.decode(enc.encode("hello world")) == "hello world"
    return enc, tiktoken


@app.cell
def _(enc):
    DEVICE = "cpu"
    LR = 1e-3
    EMBEDDING_VOCAB = enc.n_vocab
    EMBEDDING_DIM=256
    MAX_SEQ_LEN=1000
    return EMBEDDING_DIM, EMBEDDING_VOCAB, MAX_SEQ_LEN


@app.cell
def _(nn, torch):
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

    return (SelfAttentionLayer,)


@app.cell
def _(nn, torch):
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

        def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
            x = x + self.dropout(self.attn(self.norm1(x), encoder_out, encoder_out)[0])
            x = x + self.dropout(self.mlp(self.norm2(x)))
            return x


    return (CrossAttentionLayer,)


@app.cell
def _(
    CrossAttentionLayer,
    EMBEDDING_VOCAB,
    F,
    MAX_SEQ_LEN,
    SelfAttentionLayer,
    nn,
    torch,
):
    class SentenceVAE(nn.Module):
        def __init__(self, embedding_dim: int, num_heads: int, latent_dim: int, dropout = 0.1):
            super().__init__()
            # encoder
            self.embed = nn.Embedding(EMBEDDING_VOCAB, embedding_dim)
            self.pos_embed = nn.Embedding(MAX_SEQ_LEN, embedding_dim)
            self.attn1 = SelfAttentionLayer(embedding_dim, num_heads)
            self.attn2 = SelfAttentionLayer(embedding_dim, num_heads)
            self.attn3 = SelfAttentionLayer(embedding_dim, num_heads)
            self.fc_mean = nn.Linear(embedding_dim, latent_dim)
            self.fc_log_var = nn.Linear(embedding_dim, latent_dim)

            # decoder
            self.fc_in = nn.Linear(latent_dim, embedding_dim)
            self.crs_attn1 = CrossAttentionLayer(embedding_dim, num_heads)
            self.crs_attn2 = CrossAttentionLayer(embedding_dim, num_heads)
            self.crs_attn3 = CrossAttentionLayer(embedding_dim, num_heads)
            self.norm = nn.LayerNorm(embedding_dim)
            self.lm_head = nn.Linear(embedding_dim, EMBEDDING_VOCAB, bias=False)

            self.lm_head.weight = self.embed.weight


        # Returns (1, embed_dim)
        def encode(self, x: torch.Tensor) -> torch.Tensor:
            positions = torch.arange(x.size(0)).unsqueeze(1)
            x = self.embed(x) + self.pos_embed(positions)
            x = self.attn1(x)
            x = self.attn2(x)
            x = self.attn3(x)
            pooled = x.mean(dim = 1)
            mu = self.fc_mean(pooled)
            log_var = self.fc_log_var(pooled)
            return x, mu, log_var

        def decode(self, z: torch.Tensor, encoder_out) -> torch.Tensor:
            seq_len = encoder_out.size(1)
            x = self.fc_in(z).unsqueeze(1).expand(-1, seq_len, -1).clone()
            x = self.crs_attn1(x, encoder_out)
            x = self.crs_attn2(x, encoder_out)
            x = self.crs_attn3(x, encoder_out)
            return self.lm_head(self.norm(x))  # (batch, seq, vocab_size)

        def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
            if self.training:
                std = torch.exp(0.5 * log_var)
                return mu + torch.rand_like(std) * std
            return mu

        def forward(self, x):
            encoder_out, mu, log_var = self.encode(x)
            z = self.reparameterise(mu, log_var)
            logits = self.decode(z, encoder_out)

            recon_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
            kl_loss = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())

            return logits, recon_loss, kl_loss

    return (SentenceVAE,)


@app.cell
def _(EMBEDDING_DIM, SentenceVAE):
    slf_test = SentenceVAE(EMBEDDING_DIM, 8, 256)
    return (slf_test,)


@app.cell
def _(enc, torch):
    a = torch.LongTensor(enc.encode("hello world!")).reshape(1, -1)
    v = a
    v.shape
    return (v,)


@app.cell
def _(slf_test, v):
    slf_test(v)
    return


@app.cell
def _(Dataset, tiktoken, torch):
    class SentenceDataset(Dataset):
        def __init__(self, sentences, max_len=64):
            self.enc = tiktoken.get_encoding("r50k_base")  # or "p50k_base", "r50k_base"
            self.max_len = max_len
            self.pad_token = 0  # tiktoken has no pad token, just use 0
        
            self.data = []
            for sentence in sentences:
                tokens = self.enc.encode(sentence)
                # truncate or pad to max_len
                tokens = tokens[:max_len]
                tokens += [self.pad_token] * (max_len - len(tokens))
                self.data.append(torch.tensor(tokens))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    return (SentenceDataset,)


@app.cell
def _(
    DataLoader,
    SentenceDataset,
    SentenceVAE,
    sentences,
    tiktoken,
    tokenizer,
    torch,
):
    # --- Training loop ---
    def train(model, dataloader, optimizer, device, beta=1.0):
        model.train()
        total_loss = 0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits, recon_loss, kl_loss = model(batch)
            loss = recon_loss + beta * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


    # --- KL annealing ---
    def get_beta(epoch, warmup_epochs=10):
        # gradually increase KL weight so model doesn't collapse early
        return min(1.0, epoch / warmup_epochs)


    # --- Put it all together ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = tiktoken.get_encoding("cl100k_base")

    dataset = SentenceDataset(sentences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SentenceVAE(
        vocab_size=enc.n_vocab,   # 100277 for cl100k_base
        embedding_dim=256,
        num_heads=8,
        latent_dim=64,
    ).to(device)

    dataset = SentenceDataset(sentences, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    for epoch in range(50):
        beta = get_beta(epoch, warmup_epochs=10)
        loss = train(model, dataloader, optimizer, device, beta=beta)
        scheduler.step()
        print(f"epoch {epoch+1} | loss {loss:.4f} | beta {beta:.2f}")
    return (enc,)


if __name__ == "__main__":
    app.run()
