import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pickle
import json

from utils.simple_tokenizer import tokenize
from .base import LayerNorm, Transformer, TextTransformer


class TextTransformer_LearnableToken(nn.Module):
    '''
    No text input, instead, learnable tokens embedding for fixed categories
    '''
    def __init__(self, embed_dim, vocab_size, context_length, width, heads, layers, num_class):
        super().__init__()
        self.token_embedding = nn.Parameter(torch.FloatTensor(num_class, context_length, width)) # (num_node, context_length, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.ln_final = LayerNorm(width)
        self.projection = nn.Parameter(torch.empty(width, embed_dim))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
        )
        self.init()

    def init(self):
        nn.init.normal_(self.token_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.projection, std=self.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.token_embedding.dtype

    def forward(self):

        x = self.token_embedding # (?, context_length, d_model)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x[:, 0, :]).type(self.dtype)

        x = x @ self.projection

        return x


class SentenceTransformer(nn.Module):

    def __init__(self, embed_dim, vocab_size, context_length, width, heads, layers):
        super().__init__()

        context_length = 50
        Node2Info_format = pickle.load(open("./Data/Node2Info_format.pkl", "rb"))
        bert, index = Node2Info_format["vector"], Node2Info_format["index"]
        bert = torch.from_numpy(bert)
        num_class = len(Node2Info_format["index"])

        self.token_embedding = torch.zeros((num_class, context_length, width))

        mapping_node_index = pickle.load(
            open("./Data/mapping_node_index/704_to_290.pkl", "rb"))
        verbnet_topology = pickle.load(
            open("./Data/verbnet_topology_898.pkl", "rb"))
        Father2Son, objects = verbnet_topology["Father2Son"], verbnet_topology["objects"]
        objects = np.array(objects)
        
        attn_mask  = torch.BoolTensor(num_class, heads, 1+context_length, 1+context_length)
        for i, node in enumerate(objects[mapping_node_index]):
            idx = index[node]
            if len(idx) > context_length:
                idx = idx[:context_length]
            self.token_embedding[i, :len(idx)] = bert[idx]
            attn_mask[i, :, :1+len(idx), :1+len(idx)] = False

        attn_mask = attn_mask.flatten(0, 1)
        self.token_embedding = self.token_embedding.cuda()
        self.class_embedding = nn.Parameter(torch.FloatTensor(num_class, width))

        self.ln_final = LayerNorm(width)
        self.projection = nn.Parameter(torch.empty(width, embed_dim))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            attn_mask=attn_mask,
        )
        self.init()

    def init(self):
        nn.init.normal_(self.projection, std=self.transformer.width ** -0.5)
        nn.init.normal_(self.class_embedding, std=self.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.token_embedding.dtype

    def forward(self, text):

        x = torch.cat([self.class_embedding.unsqueeze(1), self.token_embedding], dim=1)  # [batch_size, 1 + n_ctx, d_model]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[:, 0, :] @ self.projection

        return x


class Hierarchical_Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.node2info = json.load(open("./Data/Node2Info_format.json", "r"))
        self.mapping_node_index = pickle.load(
            open("./Data/mapping_node_index/704_to_290.pkl", "rb"))
        verbnet_topology = pickle.load(
            open("./Data/verbnet_topology_898.pkl", "rb"))
        Father2Son, objects = verbnet_topology["Father2Son"], verbnet_topology["objects"]
        self.objects = np.array(objects)
        self.num_class = len(self.mapping_node_index)
        self.max_tokens = 60
        self.width = 512
        self.embed_dim = 512

        self.text_net = TextTransformer(
            embed_dim=self.width,
            vocab_size=49408,
            context_length=77,
            width=self.width,
            heads=8,
            layers=12
        )

        self.class_embedding = nn.Parameter(torch.FloatTensor(self.num_class, self.width))

        self.ln_final = LayerNorm(self.width)
        self.projection = nn.Parameter(torch.empty(self.width, self.embed_dim))
        self.transformer = Transformer(
            width=self.width,
            heads=8,
            layers=12,
        )
        self.init()

        self.embed = torch.zeros((self.num_class, self.max_tokens, self.width))
        for i, node in enumerate(self.objects[self.mapping_node_index]):
            info = self.node2info[node]
            tokens = np.array(tokenize(info, context_length=77)) # (?, 77)
            random_index = np.random.choice(np.array(range(len(tokens))), self.max_tokens, replace=True)
            tokens = tokens[random_index]
            tokens = torch.from_numpy(tokens)
            with torch.no_grad():
                x = self.text_net(tokens)
            self.embed[i] = x
        self.embed = self.embed.cuda()

    def init(self):
        nn.init.normal_(self.projection, std=self.transformer.width ** -0.5)
        nn.init.normal_(self.class_embedding, std=self.transformer.width ** -0.5)

        for name, param in self.text_net.named_parameters():
            param.requires_grad_(False)
        
        state_dict = torch.load("./checkpoint/ViT-B-32.pt", map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

    def forward(self, text):
        
        x = torch.cat([self.class_embedding.unsqueeze(1), self.embed], dim=1)  # [batch_size, 1 + n_ctx, d_model]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.text_net.dtype)
        x = x[:, 0, :] @ self.projection

        return x