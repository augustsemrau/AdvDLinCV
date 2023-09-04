import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

def to_device(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'



## Attention
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, f'Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})'
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        ####################### insert code here ####################### 
        self.k_projection = nn.Linear(embed_dim, embed_dim)
        self.q_projection = nn.Linear(embed_dim, embed_dim)
        self.v_projeciton = nn.Linear(embed_dim, embed_dim)
        ################################################################

        self.o_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        batch_size, seq_length, embed_dim = x.size()
        keys    = self.k_projection(x)
        queries = self.q_projection(x)
        values  = self.v_projeciton(x)

        # Rearrange keys, queries and values 
        # from batch_size x seq_length x embed_dim to (batch_size x num_head) x seq_length x head_dim
        keys = rearrange(keys, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        queries = rearrange(queries, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        values = rearrange(values, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)

        ####################### insert code here ####################### 
        ## Compute scaled dot-product attention: (batch_size x num_head) x seq_length x seq_length
        attention = torch.matmul(queries, keys.transpose(1, 2)) * self.scale
        attention = F.softmax(attention, dim=-1)

        ## Apply attention to values: (batch_size x num_head) x seq_length x head_dim
        out = torch.matmul(attention, values)
        ################################################################

        # Rearragne output
        # from (batch_size x num_head) x seq_length x head_dim to batch_size x seq_length x embed_dim
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.num_heads, d=self.head_dim)

        assert attention.size() == (batch_size*self.num_heads, seq_length, seq_length)
        assert out.size() == (batch_size, seq_length, embed_dim)

        return self.o_projection(out)



## Transformer
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_dim=None, dropout=0.0):
        super().__init__()

        self.attention = Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        fc_hidden_dim = 4*embed_dim if fc_dim is None else fc_dim

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_out = self.attention(x)
        x = self.layernorm1(attention_out + x)
        x = self.dropout(x)
        fc_out = self.fc(x)
        x = self.layernorm2(fc_out + x)
        x = self.dropout(x)

        return x

## Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):

        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0., max_seq_len).unsqueeze(1)

        ####################### insert code here ####################### 
        div_term = torch.exp(torch.arange(0., embed_dim, 2) *
                                -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        ################################################################

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        return x + self.pe[:, :seq_length]
        #return self.dropout(x)








## Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):

        super(PositionalEmbedding, self).__init__()

        # Your implemntation
        ####################### insert code here ####################### 
        #
        ################################################################

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()


        # Your implemntation
        ####################### insert code here ####################### 
        #
        #
        # return x
        ################################################################
        

## Transformer Classifier
class TransformerClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_seq_len,
                 pos_enc='fixed', pool='cls', dropout=0.0, 
                 fc_dim=None, num_tokens=50_000, num_classes=2, 
                 
    ):
        super().__init__()

        assert pool in ['cls', 'mean', 'max']
        assert pos_enc in ['fixed', 'learnable']
        
        self.pool, self.pos_enc, = pool, pos_enc
        self.token_embedding = nn.Embedding(embedding_dim=embed_dim, num_embeddings=num_tokens)

        # Initialize cls token parameter
        ####################### insert code here ####################### 
        if self.pool == 'cls':
            pass
            #
            #
        ################################################################
        
        if self.pos_enc == 'fixed':
            self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, max_seq_len=max_seq_len)
        elif self.pos_enc == 'learnable':
            self.positional_encoding = PositionalEmbedding(embed_dim=embed_dim, max_seq_len=max_seq_len)

        transformer_blocks = []
        for i in range(num_layers):
            transformer_blocks.append(
                EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, fc_dim=fc_dim, dropout=dropout))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)



    def forward(self, x):

        tokens = self.token_embedding(x)
        batch_size, seq_length, embed_dim = tokens.size()

        # Include cls token in the input sequence
        ####################### insert code here #######################
        if self.pool == 'cls':
            pass
            #
            #
        ################################################################

        x = self.positional_encoding(tokens)
        x = self.dropout(x)
        x = self.transformer_blocks(x)

        if self.pool =='max':
            x = x.max(dim=1)[0]
        elif self.pool =='mean':
            x = x.mean(dim=1)
            
        # Get cls token
        ####################### insert code here #######################
        #
        #   
        ################################################################

        return self.classifier(x)