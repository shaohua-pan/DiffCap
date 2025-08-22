"""
    Adapted from https://github.com/GuyTevet/motion-diffusion-model/tree/main.
"""
import numpy as np
import torch
import torch.nn as nn


class DualDiff(nn.Module):
    def __init__(self, njoints_smpl, njoints_2d, nfeats_smpl, nfeats_2d, d2_dim, smpl_dim, feat_latent_dim=64, feat_ff_size=256, latent_dim=256, ff_size=1024,
                 num_layers=8, num_heads=4, dropout=0.1, activation="gelu", **kargs):
        super().__init__()
        self.njoints_smpl = njoints_smpl
        self.njoints_2d = njoints_2d
        self.nfeats_smpl = nfeats_smpl
        self.nfeats_2d = nfeats_2d
        self.latent_dim = latent_dim
        self.d2_dim = d2_dim
        self.smpl_dim = smpl_dim
        self.ff_size = ff_size
        self.feat_latent_dim = feat_latent_dim
        self.feat_ff_size = feat_ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats_smpl = self.njoints_smpl * self.nfeats_smpl
        self.input_feats_2d = self.njoints_2d * self.nfeats_2d
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.embed_input_smpl = nn.Linear(self.input_feats_smpl, self.latent_dim)
        self.embed_input_2d = nn.Linear(self.input_feats_2d, self.latent_dim)
        # todo: use the same positional encoding and timestep embedding for both inputs?
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        seqTransEncoderLayer_smpl = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                               nhead=self.num_heads,
                                                               dim_feedforward=self.ff_size,
                                                               dropout=self.dropout,
                                                               activation=self.activation)
        seqTransEncoderLayer_2d = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                             nhead=self.num_heads,
                                                             dim_feedforward=self.ff_size,
                                                             dropout=self.dropout,
                                                             activation=self.activation)
        self.seqTransEncoder_smpl = nn.TransformerEncoder(seqTransEncoderLayer_smpl, num_layers=self.num_layers)
        self.seqTransEncoder_2d = nn.TransformerEncoder(seqTransEncoderLayer_2d, num_layers=self.num_layers)
        # smplEncoderLayer = nn.TransformerEncoderLayer(d_model=self.feat_latent_dim,
        #                                               nhead=self.num_heads,
        #                                               dim_feedforward=self.feat_ff_size,
        #                                               dropout=self.dropout,
        #                                               activation=self.activation)
        # self.smplTransEncoder = nn.TransformerEncoder(smplEncoderLayer, num_layers=self.num_layers)
        # d2EncoderLayer = nn.TransformerEncoderLayer(d_model=self.feat_latent_dim,
        #                                             nhead=self.num_heads,
        #                                             dim_feedforward=self.feat_ff_size,
        #                                             dropout=self.dropout,
        #                                             activation=self.activation)
        # self.d2TransEncoder = nn.TransformerEncoder(d2EncoderLayer, num_layers=self.num_layers)
        self.embed_2d = nn.Linear(self.d2_dim, self.latent_dim)
        self.embed_smpl = nn.Linear(self.smpl_dim, self.latent_dim)
        self.embed_output_smpl = nn.Linear(self.latent_dim, self.input_feats_smpl)
        self.embed_output_2d = nn.Linear(self.latent_dim, self.input_feats_2d)

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs,
                                                                                                  1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, nframes], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        assert isinstance(x, tuple)
        x_dot = x[1]
        x = x[0]
        bs, njoints_2d, nfeats_2d, nframes = x.shape
        bs, njoints_smpl, nfeats_smpl, nframes = x_dot.shape
        time_emb = self.embed_timestep(timesteps)  # [1, bs, d]
        # pose_feat = self.smplTransEncoder(x_dot.clone().permute(3, 0, 1, 2).reshape(nframes, bs, njoints_smpl * nfeats_smpl))
        d2_emb = self.embed_2d(x.clone().reshape(bs, nframes * njoints_2d * nfeats_2d))
        d2_feat = self.mask_cond(d2_emb).unsqueeze(0)  # [1, bs, d]
        d2_feat += time_emb
        smpl_emb = self.embed_smpl(x_dot.clone().reshape(bs, nframes * njoints_smpl * nfeats_smpl))
        smpl_feat = self.mask_cond(smpl_emb).unsqueeze(0)  # [1, bs, d]
        smpl_feat += time_emb
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints_2d * nfeats_2d)
        x = self.embed_input_2d(x)  # [nframes, bs, d]
        x_dot = x_dot.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints_smpl * nfeats_smpl)
        x_dot = self.embed_input_smpl(x_dot)  # [nframes, bs, d]
        xseq = torch.cat((d2_feat, x), dim=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder_2d(xseq)[1:]  # [seqlen, bs, d]
        output = self.embed_output_2d(output).reshape(nframes, bs, njoints_2d, nfeats_2d)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        xseq_dot = torch.cat((smpl_feat, x_dot), dim=0)  # [seqlen+1, bs, d]
        xseq_dot = self.sequence_pos_encoder(xseq_dot)  # [seqlen+1, bs, d]
        output_dot = self.seqTransEncoder_smpl(xseq_dot)[1:]  # [seqlen, bs, d]
        output_dot = self.embed_output_smpl(output_dot).reshape(nframes, bs, njoints_smpl, nfeats_smpl)
        output_dot = output_dot.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output, output_dot


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

#
# encoder_layer = nn.TransformerEncoderLayer(d_model=8, nhead=8)
# transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
# transformer_encoder.eval()
# src = torch.rand(1, 1, 8)
# src2 = torch.rand(1, 1, 8)
# src11 = torch.cat((src2, src), dim=1)
# src11 = src11.permute(1, 0, 2)
# mask = torch.zeros(8, 2, 2)
# mask[:, :, 0] = True
# mask = mask > 0
# # mask[:, :, :] = 0
# out1 = transformer_encoder(src11, mask=mask)
# print(out1)
# src2 = torch.rand(1, 1, 8)
# src22 = torch.cat((src2, src), dim=1)
# src22 = src22.permute(1, 0, 2)
# out = transformer_encoder(src22, mask=mask)
# print(out)
# print(out1 == out)
