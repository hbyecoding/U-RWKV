import torch
import torch.nn as nn
from retention import MultiScaleRetention

class RetNet(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, double_v_dim=False):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

        # Initialize multi-scale retention layers
        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])

        # Initialize feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])

        # Initialize layer normalizations
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
    
    def forward(self, X):
        """
        Standard forward pass.
        X: (batch_size, sequence_length, hidden_size)
        """
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y

        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        Recurrent forward pass for step-by-step processing.
        x_n: (batch_size, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        n: current time step
        """
        s_ns = []
        for i in range(self.layers):
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n
        
        return x_n, s_ns
    
    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        Chunkwise forward pass for processing chunks of sequences.
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        i: current chunk index
        """
        r_is = []
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
            y_i = o_i + x_i
            r_is.append(r_i)
            x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i
        
        return x_i, r_is
    
    
if __name__ == "__main__":
#     x = torch.rand((1, 3, 256, 256)).cuda()
#     model = LoRA_4_5().cuda()
#     y1, y2 = model(x)
#     print(y1.shape, y2.shape)
#     print(count_params(model))
#     print("END")

    # import os 
    # os.environ['CUDA_VISIBLE_DEVICES']='7'

    import time 
    from thop import profile, clever_format
    
    '''
    !!!!!!!!
    Caution: Please comment out the code related to reparameterization and retain only the 5x5 convolutional layer in the OmniShift.
    !!!!!!!!
    '''
    # Test parameters
    batch_size = 1
    sequence_length = 256
    hidden_dim = 256
    ffn_size = 512
    heads = 8
    layers = 4
    double_v_dim = False
    
    # x=torch.zeros((1, 3, 256, 256)).type(torch.FloatTensor).cuda()

    x = torch.zeros((batch_size, sequence_length, hidden_dim)).type(torch.FloatTensor).cuda()
    model = RetNet(layers, hidden_dim, ffn_size=ffn_size, heads=heads)
    model.cuda()
    
    # model = LoRA_4_5()
    # model = LoRA__5() 
    # model = LoRA_dw_rwkv_first_4_5()
    # model = LoRA_dw_4_5()
    model = RetNet(layers,hidden_dim,ffn_size=ffn_size,heads=heads)
    # x=torch.zeros((1, 768,32,32)).type(torch.FloatTensor).cuda()
    # model = BinaryOrientatedRWKV2D(n_embd=768, n_layer=12)
    model.cuda() 
    
    since = time.time()
    y=model(x)
    print("time", time.time()-since)
    
    flops, params = profile(model, inputs=(x, ))  
    flops, params = clever_format([flops, params], '%.6f')
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")    