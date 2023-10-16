from transformers import RobertaPreLayerNormConfig, RobertaPreLayerNormModel

class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()

        self.n_classes = len(
            [
                'background',
                'liver',
                'spleen',
                'left kidney',
                'right kidney',
                'bowel'
            ])
        in_chans = 1
        self.encoder = timm.create_model(
            'regnety_002',
            pretrained=False,
            features_only=True,
            in_chans=in_chans,
        )
        encoder_channels = tuple(
            [in_chans]
            + [
                self.encoder.feature_info[i]["num_chs"]
                for i in range(len(self.encoder.feature_info))
            ]
        )
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=self.n_classes,
            activation=None,
            kernel_size=3,
        )

        self.bce_seg = nn.BCEWithLogitsLoss()

    def forward(self, x_in):
        enc_out = self.encoder(x_in)

        decoder_out = self.decoder(*[x_in] + enc_out)
        x_seg = self.segmentation_head(decoder_out)

        return nn.Sigmoid()(x_seg)

class FeatureExtractor(nn.Module):
    def __init__(self, hidden, num_channel):
        super(FeatureExtractor, self).__init__()

        self.hidden = hidden
        self.num_channel = num_channel

        self.cnn = timm.create_model(model_name = 'regnety_002',
                                     pretrained = True,
                                     num_classes = 0,
                                     in_chans = num_channel)

        self.fc = nn.Linear(hidden, hidden//2)

    def forward(self, x):
        batch_size, num_frame, h, w = x.shape
        x = x.reshape(batch_size, num_frame//self.num_channel, self.num_channel, h, w)
        x = x.reshape(-1, self.num_channel, h, w)
        x = self.cnn(x)
        x = x.reshape(batch_size, num_frame//self.num_channel, self.hidden)

        x = self.fc(x)
        return x

class ContextProcessor(nn.Module):
    def __init__(self, hidden):
        super(ContextProcessor, self).__init__()
        self.transformer = RobertaPreLayerNormModel(
            RobertaPreLayerNormConfig(
                hidden_size = hidden//2,
                num_hidden_layers = 1,
                num_attention_heads = 4,
                intermediate_size = hidden*2,
                hidden_act = 'gelu_new',
                )
            )

        del self.transformer.embeddings.word_embeddings

        self.dense = nn.Linear(hidden, hidden)
        self.activation = nn.ReLU()


    def forward(self, x):
        x = self.transformer(inputs_embeds = x).last_hidden_state

        apool = torch.mean(x, dim = 1)
        mpool, _ = torch.max(x, dim = 1)
        x = torch.cat([mpool, apool], dim = -1)

        x = self.dense(x)
        x = self.activation(x)
        return x
        

class Custom3DCNN(nn.Module):
    def __init__(self, hidden = 368, num_channel = 2):
        super(Custom3DCNN, self).__init__()

        self.full_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)
        self.kidney_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)
        self.liver_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)
        self.spleen_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)

        self.full_processor = ContextProcessor(hidden=hidden)
        self.kidney_processor = ContextProcessor(hidden=hidden)
        self.liver_processor = ContextProcessor(hidden=hidden)
        self.spleen_processor = ContextProcessor(hidden=hidden)

        self.bowel = nn.Linear(hidden, 2)
        self.extravasation = nn.Linear(hidden, 2)
        self.kidney = nn.Linear(hidden, 3)
        self.liver = nn.Linear(hidden, 3)
        self.spleen = nn.Linear(hidden, 3)

        self.softmax = nn.Softmax(dim = -1)

    def forward(self, full_input, crop_liver, crop_spleen, crop_kidney, mask, mode):
        full_output = self.full_extractor(full_input)
        kidney_output = self.kidney_extractor(crop_kidney)
        liver_output = self.liver_extractor(crop_liver)
        spleen_output = self.spleen_extractor(crop_spleen)

        full_output2 = self.full_processor(torch.cat([full_output, kidney_output, liver_output, spleen_output], dim = 1))
        kidney_output2 = self.kidney_processor(torch.cat([full_output, kidney_output], dim = 1))
        liver_output2 = self.liver_processor(torch.cat([full_output, liver_output], dim = 1))
        spleen_output2 = self.spleen_processor(torch.cat([full_output, spleen_output], dim = 1))

        bowel = self.bowel(full_output2)
        extravasation = self.extravasation(full_output2)
        kidney = self.kidney(kidney_output2)
        liver = self.liver(liver_output2)
        spleen = self.spleen(spleen_output2)


        any_injury = torch.stack([
            self.softmax(bowel)[:, 0],
            self.softmax(extravasation)[:, 0],
            self.softmax(kidney)[:, 0],
            self.softmax(liver)[:, 0],
            self.softmax(spleen)[:, 0]
        ], dim = -1)
        any_injury = 1 - any_injury
        any_injury, _ = any_injury.max(1)
        return bowel, extravasation, kidney, liver, spleen, any_injury
