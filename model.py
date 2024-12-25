import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class CSABlock(nn.Module):
    """Causal self-attention block"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # K, Q, V projections for multiple heads
        ### YOUR CODE HERE ###
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.key_proj = nn.Linear(config.n_embd, config.n_embd)
        self.query_proj = nn.Linear(config.n_embd, config.n_embd)
        self.value_proj = nn.Linear(config.n_embd, config.n_embd)
        ### YOUR CODE HERE ###
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # mask for causal attention
        ### YOUR CODE HERE ###
        #self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.mask = self.mask.view(1, 1, config.block_size, config.block_size)  # Ensure correct shape
        ### YOUR CODE HERE ###

        #self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, L, C = x.size()

        # Q, K, V for all heads
        ### YOUR CODE HERE ###
        queries = self.query_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        keys = self.key_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        values = self.value_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        ### YOUR CODE HERE ###

        # Causal self-attention
        # attention dropout
        ### YOUR CODE HERE ###
        attn_weights = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = attn_weights.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))  # Apply causal mask
        attn_save =  attn_weights.clone().detach()  # Save attention weights without dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        ### YOUR CODE HERE ###

        # Apply the attention to the values; Combine all head outputs
        ### YOUR CODE HERE ###
        attn_output = torch.matmul(attn_weights, values)  # Compute the attention output
        y = attn_output.transpose(1, 2).reshape(B, L, self.n_embd)
        ### YOUR CODE HERE ###

        # Readout projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save # attn_save is the attention mask without dropout

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CSABlock(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        y, attn = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.padding_token_id = 0
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformers
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.isconditional = config.isconditional

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        Optimizer and learning rate scheduler
        """

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias') or ('bias' in pn):
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add('pos_emb')
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, condition_split_id=None):
        b, t = idx.size()
        assert t <= self.block_size


        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        type_embeddings = self.type_emb(torch.ones((b,t), dtype = torch.long, device = idx.device))
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)

        attn_maps = []

        for layer in self.blocks:
            x, attn = layer(x)
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            mask = targets != self.padding_token_id
            if self.isconditional:
                # Create a range tensor [0, 1, 2, ..., seq_len-1]
                range_tensor = torch.arange(t, device=mask.device).expand(b, -1)
                # Expand split_id to match the shape of range_tensor for broadcasting
                expanded_split_id = condition_split_id.unsqueeze(1).expand(-1, t)
                # Generate the update mask (True where range_tensor < expanded_split_id)
                cond_mask = range_tensor < expanded_split_id
                # Update the original mask (set to False where the condition is False)
                mask[cond_mask] = False
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            loss = (loss * mask.view(-1)).sum() / mask.sum()

        return logits, loss, attn_maps