import numpy as np
import torch
import torch.nn as nn

class DiffusionTran3dCond(nn.Module):
    def __init__(self, njoints, nfeats, imu_3d_dim, mp_dim,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", **kargs):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.imu_3d_dim = imu_3d_dim
        self.mp_dim = mp_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = self.njoints * self.nfeats
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.embed_input = nn.Linear(self.input_feats + self.imu_3d_dim, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_mp = nn.Linear(self.mp_dim, self.latent_dim)
        self.embed_output = nn.Linear(self.latent_dim, self.input_feats)

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, nframes], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        # mp_emb = self.embed_mp(y['mp'])
        # emb += self.mask_cond(mp_emb)

        x = x.permute(0, 3, 1, 2).reshape(bs * nframes, njoints * nfeats)  # [bs*nframes, njoints*nfeats]
        imu_3d = y['3d_imu'].reshape(bs * nframes, self.imu_3d_dim)
        x = torch.cat((x, imu_3d), dim=1)  # [bs*nframes, njoints*nfeats+mp_imu_dim]
        x = self.embed_input(x).reshape(bs, nframes, -1).permute(1, 0, 2)  # [nframes, bs, d]

        xseq = torch.cat((emb, x), dim=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq)[1:]  # [seqlen, bs, d]

        output = self.embed_output(output).reshape(nframes, bs, njoints, nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
