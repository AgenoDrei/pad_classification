import torch
from torch import nn


class RetinaNet(nn.Module):
    def __init__(self, frame_stump, num_frames=20, do_avg_pooling=True):
        super(RetinaNet, self).__init__()
        self.stump = frame_stump
        self.pool_stump = do_avg_pooling
        self.num_frames = num_frames
        self.pool_params = (self.stump.last_linear.in_features, 256) if self.pool_stump else (
            98304, 1024)  # fix for higher resolutions / different networks
        self.out_stump = self.pool_params[0]

        self.avg_pooling = self.stump.avg_pool
        self.temporal_pooling = nn.MaxPool1d(self.num_frames, stride=1, padding=0, dilation=self.out_stump)

        self.after_pooling = nn.Sequential(nn.Linear(self.out_stump, self.pool_params[1]), nn.ReLU(), nn.Dropout(p=0.5),
                                           nn.Linear(self.pool_params[1], 2))
        # self.fc1 = nn.Linear(self.out_stump, 256)
        # self.fc2 = nn.Linear(256, 2)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        features = []
        for idx in range(0, x.size(1)):  # Iterate over time dimension
            out = self.stump.features(x[:, idx, :, :, :])  # Shove batch trough stump
            out = self.avg_pooling(out) if self.pool_stump else out
            out = out.view(out.size(0), -1)  # Flatten results for fc
            features.append(out)  # Size: (B, c*h*w)
        out = torch.cat(features, dim=1)
        out = self.temporal_pooling(out.unsqueeze(dim=1))
        out = self.after_pooling(out.view(out.size(0), -1))
        return out


class RetinaNet2(nn.Module):
    def __init__(self, frame_stump, num_frames=20, pooling_strategy='max'):
        super(RetinaNet2, self).__init__()
        self.stump = frame_stump
        self.pooling_strategy = pooling_strategy
        self.pool_params = (self.stump.last_linear.in_features, 256) if self.pool_stump else (
            98304, 1024)  # fix for higher resolutions / different networks
        self.out_stump = self.pool_params[0]

        self.pooling = self.stump.avg_pool if pooling_strategy == 'avg' else nn.MaxPool2d(self.stump.avg_pool.kernel_size, stride=self.stump.avg_pool.stride)
        self.after_pooling = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.out_stump, self.pool_params[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.pool_params[1], 2))

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        x = self.stump.features(x)
        h = self.pooling(x)
        h = h.view(h.size(0), -1)  # Flatten results for fc / pooling
        h = torch.max(h, 0, keepdim=True)[0]  # Temproal pooling over 1 dim
        out = self.after_pooling(h)
        return out


class BagNet(nn.Module):
    def __init__(self, stump, num_attention_neurons=128, attention_strategy='normal', pooling_strategy='avg', stump_type='alexnet'):
        super(BagNet, self).__init__()
        self.stump = stump
        self.stump_type = stump_type
        self.attention_strategy = attention_strategy
        self.L = 4096 if stump_type == 'alexnet' else 1536  # FC layer size of AlexNet / Inception
        self.D = num_attention_neurons
        self.K = 1  # Just why, paper, whyyyy? -> Vector reasons, maybe?
        self.feature_extractor_part1 = stump.features
        self.pool, self.num_features = self._get_pooling_params(pooling_strategy)  
        #self.feature_extractor_part2 = nn.Sequential(
        #    nn.Dropout(),
        #    nn.Linear(self.num_features, self.L),  # 256: Channel size of AlexNet features
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #    nn.Linear(self.L, self.L),
        #    nn.ReLU(inplace=True),
        #)
        self.feature_extractor_part2 = None
        if stump_type == 'alexnet':
             self.feature_extractor_part2 = stump.classifier[:-1]
        elif stump_type == 'inception':
            stump.last_linear = nn.Identity()
            self.feature_extractor_part2 = stump.avg_pool if pooling_strategy == 'avg' else nn.MaxPool2d(stump.avg_pool.kernel_size, stride=stump.avg_pool.stride)
        self.attention, self.att_v, self.att_u, self.att_weights = self._get_attention_net(attention_strategy)
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        if self.stump_type == 'alexnet':
            H = self.pool(H)
            H = H.view(-1, self.num_features)
        H = self.feature_extractor_part2(H)  # N x L, Number of bag elements
        if self.stump_type == 'inception':
            H = H.view(-1, 1536)

        if self.attention_strategy == 'normal':
            A = self.attention(H)  # N x K
        else:
            A_V = self.att_v(H)  # NxD
            A_U = self.att_u(H)  # NxD
            A = self.att_weights(A_V * A_U)  # element wise multiplication # NxK

        A = torch.transpose(A, 1, 0)  # K x N
        A = nn.functional.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # K x L, multiply attention weights with bag elements
        y_prob = self.classifier(M)
        y_pred = torch.ge(y_prob, 0.5).float()
        return y_prob, y_pred, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

    def _get_pooling_params(self, strategy='avg'):
        if strategy == 'avg':
            return nn.AdaptiveAvgPool2d((6, 6)), (256 * 6 * 6)
        elif strategy == 'max':
            return nn.AdaptiveMaxPool2d((6, 6)), (256 * 6 * 6)
        else:
            return nn.Identity(), (256 * 11 * 11)

    def _get_attention_net(self, strategy='normal'):
        att = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        att_v = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        att_u = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        att_weights = nn.Linear(self.D, self.K)
        return (att, nn.Identity(), nn.Identity(), nn.Identity()) if strategy == 'normal' else (nn.Identity(), att_v, att_u, att_weights)

