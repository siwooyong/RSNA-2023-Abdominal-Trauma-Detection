class RSNAClassifier(nn.Module):
    def __init__(self, model_arch, hidden_dim=128, seq_len=3, pretrained=False):
        super().__init__()
        self.seq_len = seq_len
        self.model_arch = model_arch
        self.model = timm.create_model(model_arch, in_chans=3, pretrained=pretrained)


        cnn_feature = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)


        self.spatialdropout = SpatialDropout(CFG.dropout)
        self.gru = nn.GRU(cnn_feature, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.mlp_attention_layer = MLPAttentionNetwork(2 * hidden_dim)
        self.logits = nn.Sequential(
            nn.Linear(2 * hidden_dim, 13),
        )
    def forward(self, x):
        bs = x.size(0)
        x = x.reshape(bs*self.seq_len//3, 3, x.size(2), x.size(3))
        features = self.model(x)
        features = self.pooling(features).view(bs*self.seq_len//3, -1)
        features = self.spatialdropout(features) 
        # print(features.shape)
        features = features.reshape(bs, self.seq_len//3, -1) 
        features, _ = self.gru(features)            
        atten_out = self.mlp_attention_layer(features) 
        pred = self.logits(atten_out)
        pred = pred.view(bs, -1)
        return pred
