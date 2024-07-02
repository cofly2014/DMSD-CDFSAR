import torch.nn as nn
from torch import einsum
import torch.nn.functional as F

import random
from einops import rearrange
import os
import torch
from utils import getcombinations, EuclideanDistance, Euclidean_Distance
import copy
from models.resNet import MyResNet


class PreNormattention_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm(q), self.norm(k), self.norm(v), **kwargs) + q


class Transformer_v1(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte=0.05, mlp_dim=2048,
                 dropout_ffn=0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                    # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                    PreNormattention_qkv(dim,
                                         Attention_qkv(dim, heads=heads, dim_head=dim_head_k, dropout=dropout_atte)),
                    FeedForward(dim, mlp_dim, dropout=dropout_ffn),
                ]))

    def forward(self, q, k, v):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(q, k, v)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x, x, x)
                x = ff(x) + x
        return x


class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class Attention_qkv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        bk = k.shape[0]
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', b=bk, h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', b=bk, h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)  # [30, 8, 8, 5]

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

####################################################################

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1, -2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len]
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

        # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(
                - cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        # cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(
                - cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]


def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len]
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

        # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(
                - cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        # cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(
                - cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]


class CROSS_DOMAIN_FSAR(nn.Module):
    """
    OTAM with a CNN backbone.
    """

    def __init__(self, args):
        super(CROSS_DOMAIN_FSAR, self).__init__()
        self.argss = args

        self.argss.num_patches = 16
        self.argss.reduction_fac = 4

        self.mid_dim = 2048
        self.mid_dim = 512
        self.mid_layer = nn.Sequential()

        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(1.0)

        self.student_encoder = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.argss.TRANSFORMER_DEPTH))
        self.inner_teacher = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.argss.TRANSFORMER_DEPTH))
        self.decoder = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.argss.TRANSFORMER_DEPTH))


        self.classification_layer = nn.Sequential(
            nn.LayerNorm(self.mid_dim),
            nn.Linear(self.mid_dim, self.argss.class_num)
        )

        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.copyStudent = True
        self.backbone = MyResNet(self.argss)
        self.start_cross = self.argss.start_cross

    '''
    计算一个全序列样本，和改样本子序列之间判别概率的KL散度
    '''

    def self_seq_loss(self, P_local, P_global, bs):
        self_loss = sum([-torch.sum(torch.mul(torch.log(P_local[index]), P_global[index])) for index in range(bs)])
        return self_loss

    '''
    计算一个全序列样本，和该样本所属的类，的其他子序列之间判别概率的KL散度
    '''

    def cross_seq_loss(self, P_local, P_global, inputs, bs):
        shuffle_P_local = []
        target_label_dict = {}
        for i, v in enumerate(inputs["target_labels"]):
            if target_label_dict.get(v.item()):
                target_label_dict[v.item()].append(i)
            else:
                target_label_dict[v.item()] = [i]
        for i, value in enumerate(inputs["target_labels"]):
            # 提取classlabel相同的，获取index，index不同继续抽
            while True:
                mid = random.choice(target_label_dict[value.item()])
                if int(i) != mid:
                    break
                else:
                    pass
            shuffle_P_local.append(P_local[(mid):((mid + 1))])

        shuffle_P_local = torch.concat(shuffle_P_local, dim=0)
        cross_loss = sum([-torch.sum(torch.mul(torch.log(shuffle_P_local[index]), P_global[index])) for index in range(bs)])
        return cross_loss

    def get_feats(self, image_features):
        features = self.backbone(image_features)[-1]
        shape = features.shape
        features = features.reshape(int(shape[0] / self.argss.seq_len), self.argss.seq_len,  shape[1], shape[2], shape[3])
        features = features.permute(0, 2, 1, 3, 4)
        features = self.avgpool(features).squeeze().permute(0,2,1)
        return features

    def enhance_source(self, support_features, target_domain_features):
        b_s, seq_size, d_size = target_domain_features.shape
        # 第一维度是池子中的视频帧数量, 例如 25*8, 1024
        target_domain_features = target_domain_features.reshape(b_s * seq_size, d_size)
        b_s, seq_size, d_size = support_features.shape
        enhanced_support_features = support_features.clone()
        #support_features = [x for x in support_features]
        for i, support_feature in enumerate(support_features):
            # 8 1024    25*8 1024  =>  8*25,  8
            s2t_sim = cos_sim(support_feature, target_domain_features)
            _, indices = s2t_sim.topk(1, dim=1, largest=True, sorted=True)
            column = indices.squeeze(-1)
            enhanced_support_features[i] =  0.6*enhanced_support_features[i] + 0.4*target_domain_features[ indices.squeeze(-1)]
            #给最大值出现的行全部置零
        return enhanced_support_features

    def forward(self, inputs, iteration=1):  # 获得support support labels, query, support real class
        support_images, support_labels, target_images, support_real_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels']  # [200, 3, 224, 224] inputs["real_support_labels"]
        target_domain_set = inputs['target_domain_set']
        #######################################################################################################
        # 获得源数据和目标数据的backbone的特征
        support_features = self.get_feats(support_images)
        query_features = self.get_feats(target_images)
        target_domain_features = self.get_feats(target_domain_set)
        source_domain_features = torch.concat([support_features, query_features], dim=0)
        support_num = support_features.shape[0]
        #######################################################################################################
        #在backbone获取的特征中，用目标域来增强源域
        #对每个无标签的目标域的样本用temporal transformer进行编码
        target_domain_features_enc = self.student_encoder(target_domain_features, target_domain_features, target_domain_features)
        ######################################################################################
        # 通过student网络获得source_domain 中support和query的特征，其中support_features_g是原型
        source_domain_features_mean_s = self.recalibration_centor(source_domain_features)
        source_domain_features_mean_s = source_domain_features_mean_s.unsqueeze(0).expand(support_num, -1, -1)
        #query_features_g = self.student_encoder(source_domain_features_mean_s, query_features, query_features)
        #support_features_g = self.student_encoder(source_domain_features_mean_s, support_features, support_features)
        query_features_g = self.student_encoder(query_features, query_features, query_features)
        support_features_g = self.student_encoder(support_features, support_features, support_features)
        unique_labels = torch.unique(support_labels)
        support_features_g_pro = [
            torch.mean(torch.index_select(support_features_g, 0, extract_class_indices(support_labels, c)), dim=0)
            for c in unique_labels]
        support_features_g_pro = torch.stack(support_features_g_pro)


        source_domain_features_g = torch.concat([support_features_g, query_features_g], dim=0)


        cum_dist_g = -self.otam_distance(support_features_g_pro, query_features_g)  # 全局对比全局
        P_global = F.softmax(cum_dist_g, dim=-1)
        # 监督学习，对用backbone中提取的源域的数据进行分类
        class_logits = self.classification_layer(source_domain_features_g.mean(1))
        ##########################################################################################
        ##########################################################################################
        #############
        #源域循环一致性
        reconstruct_source_domain_features = self.decoder(source_domain_features_g, source_domain_features_g, source_domain_features_g)
        reconstruct_diff_source = reconstruct_source_domain_features - source_domain_features_g
        reconstruct_norm_sq_source = torch.sqrt(torch.norm(reconstruct_diff_source, dim=[-2, -1]) / (512*8))
        reconstruct_norm_distance_source = torch.mean(reconstruct_norm_sq_source)  # 重建误差
        #############
        #目标域循环一致性
        reconstruct_target_domain_features =self.decoder(target_domain_features_enc, target_domain_features_enc, target_domain_features_enc)
        reconstruct_diff_target = reconstruct_target_domain_features - target_domain_features
        reconstruct_norm_target = torch.sqrt(torch.norm(reconstruct_diff_target, dim=[-2, -1]) / (512*8))
        reconstruct_norm_distance_target = torch.mean(reconstruct_norm_target)  # 重建误差
        #############
        reconstruct_norm_distance = (reconstruct_norm_distance_source + reconstruct_norm_distance_target)/2
        #reconstruct_norm_distance = reconstruct_norm_distance_target
        print("reconstruct_norm_distance is {}".format(reconstruct_norm_distance))
        ##########################################################################################################

        meta_kl_loss = 0
        superviesed_kl_loss = 0
        center_loss = 0
        if iteration > self.start_cross:
            if self.copyStudent:
                self.inner_teacher = copy.deepcopy(self.student_encoder)
                for param_t in self.inner_teacher.parameters():
                    param_t.requires_grad = False
                self.copyStudent = False
            print("start to do the knowledge distillation...")
            #进行投影中心的计算，第一行是用均值的方法，第二个是用一样本相似度加权的方法。
            target_domain_features_mean_s = torch.mean(target_domain_features, dim=0).unsqueeze(0).expand(support_num, -1, -1)

            #target_domain_features_mean_s = self.recalibration_centor(target_domain_features)
            target_domain_features_mean_s = torch.mean(target_domain_features, dim=0)
            target_domain_features_mean_s = target_domain_features_mean_s.unsqueeze(0).expand(support_num, -1, -1)
            teacher_enhanced_source_support_features = 0.5 * (self.inner_teacher(target_domain_features_mean_s, support_features, support_features) + support_features)
            query_num = query_features.shape[0]
            target_domain_features_mean_q = torch.mean(target_domain_features, dim=0).unsqueeze(0).expand(query_num, -1, -1)
            teacher_enhanced_source_query_features = 0.5 * (self.inner_teacher(target_domain_features_mean_q, query_features, query_features) + query_features)

            unique_labels = torch.unique(support_labels)
            teacher_enhanced_source_support_features_pro = [
                torch.mean(torch.index_select(teacher_enhanced_source_support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
            teacher_enhanced_source_support_features_pro = torch.stack(teacher_enhanced_source_support_features_pro)


            '''
            teacher_enhanced_source_support_features_pro, teacher_enhanced_source_support_features, teacher_enhanced_source_query_features = \
                self.temporal_transformer(teacher_enhanced_source_support_features, teacher_enhanced_source_query_features, support_labels)
            '''
            teacheer_source_domain_features_g = torch.concat([teacher_enhanced_source_support_features, teacher_enhanced_source_query_features], dim=0)
            cum_dist_teacher = -self.otam_distance(teacher_enhanced_source_support_features_pro, teacher_enhanced_source_query_features)  # 全局对比全局
            P_teacher = F.softmax(cum_dist_teacher, dim=-1)

            teacher_class_logits = self.classification_layer(teacheer_source_domain_features_g.mean(1))


            meta_kl_loss = F.kl_div(torch.log(P_teacher), P_global, reduction='sum')
            class_logits_p = torch.softmax(class_logits, dim=-1)
            teacher_class_logits_p = torch.softmax(teacher_class_logits, dim=-1)
            superviesed_kl_loss =  F.kl_div(torch.log(teacher_class_logits_p), class_logits_p, reduction='sum')

            #在每个episode中将target center和source center拉近
            center_loss = source_domain_features_mean_s[0] - target_domain_features_mean_s[0]
            center_loss = torch.sqrt(torch.norm(center_loss, dim=[0,1]) / (512 * 8))

            with torch.no_grad():
                m = 0.99980
                # 用student网络参数，动量跟新inner_teacher网络
                for param_q, param_k in zip(self.student_encoder.parameters(), self.inner_teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


        # (1) metric distance rest+otam 用来对比测试
        unique_labels = torch.unique(support_labels)
        support_features_pro = [
            torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
            for c in unique_labels]
        support_features_pro = torch.stack(support_features_pro)
        cum_dist_origin = -self.otam_distance(support_features_pro, query_features)
        ##########################################################################################################
        return_dict = {
            'class_logits': class_logits,  #supervised loss
            'meta_logits': cum_dist_g,
            "meta_kl_loss": meta_kl_loss,
            "superviesed_kl_loss": superviesed_kl_loss,
            "reconstruct_distance": reconstruct_norm_distance
        }  # [5， 5] , [10 64]

        return return_dict

    def forward_test(self, inputs, iteration=1):  # 获得support support labels, query, support real class
        support_images, support_labels, target_images, support_real_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels']  # [200, 3, 224, 224] inputs["real_support_labels"]
        #######################################################################################################
        # 获得源数据和目标数据的backbone的特征
        support_features = self.get_feats(support_images)
        query_features = self.get_feats(target_images)
        source_domain_features = torch.concat([support_features, query_features], dim=0)
        support_num = support_features.shape[0]
        query_num = query_features.shape[0]
        #######################################################################################################

        ######################################################################################
        # 通过student网络获得source_domain 中support和query的特征，其中support_features_g是原型
        #support_features_g_pro, support_features_g, query_features_g = self.temporal_transformer(support_features, query_features, support_labels)
        #

        source_domain_features_mean_s = self.recalibration_centor(source_domain_features)
        source_domain_features_mean_s = source_domain_features_mean_s.unsqueeze(0).expand(support_num, -1, -1)
        #support_features_g = self.student_encoder(source_domain_features_mean_s, support_features, support_features)
        support_features_g = self.student_encoder(support_features, support_features, support_features)

        source_domain_features_mean_q = self.recalibration_centor(source_domain_features)
        source_domain_features_mean_q = source_domain_features_mean_q.unsqueeze(0).expand(query_num, -1, -1)
        #query_features_g = self.student_encoder(source_domain_features_mean_q, query_features, query_features)
        query_features_g = self.student_encoder(query_features, query_features, query_features)

        unique_labels = torch.unique(support_labels)
        support_features_g_pro = [
            torch.mean(torch.index_select(support_features_g, 0, extract_class_indices(support_labels, c)), dim=0)
            for c in unique_labels]
        support_features_g_pro = torch.stack(support_features_g_pro)


        source_domain_features_g = torch.concat([support_features_g, query_features_g], dim=0)


        cum_dist_g = -self.otam_distance(support_features_g_pro, query_features_g)  # 全局对比全局
        P_global = F.softmax(cum_dist_g, dim=-1)
        # 监督学习，对用backbone中提取的源域的数据进行分类
        class_logits = self.classification_layer(source_domain_features_g.mean(1))
        ##########################################################################################

        return_dict = {
            'dis_logits': cum_dist_g
        }  # [5， 5] , [10 64]

        return return_dict

    #Recalibration target center
    def recalibration_centor(self, target_domain_features):
        target_num = target_domain_features.shape[0]
        consimilirity_list = []
        for i in range(target_num-1):
            target_domain_features_roll = torch.roll(target_domain_features, i+1, dims=0)
            con_similirity = cos_sim(target_domain_features, target_domain_features_roll)
            con_similirity = torch.mean(torch.mean(torch.exp(con_similirity), dim=-1), dim=-1)
            consimilirity_list.append(con_similirity)
        consimilirity_all = torch.stack(consimilirity_list, dim=1)
        instances_similairy = torch.mean(consimilirity_all, dim=1)
        result = 0
        for i in range(target_num):
            result = result + instances_similairy[i]*target_domain_features[i]
        return result/target_num

    def temporal_transformer(self, support_features, query_features, support_labels):
        query_features = self.student_encoder(query_features, query_features, query_features)
        support_features = self.student_encoder(support_features, support_features, support_features)
        unique_labels = torch.unique(support_labels)
        support_features_pro = [
            torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
            for c in unique_labels]
        support_features_pro = torch.stack(support_features_pro)
        return support_features_pro, support_features, query_features

    def proto_distance(self, support_features, target_features):
        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]
        support_features = rearrange(support_features, 'b s d -> (b s) d')  # 5 8 1024-->40  1024
        target_features = rearrange(target_features, 'b s d -> (b s) d')
        frame_sim = cos_sim(target_features, support_features)  # 类别数量*每个类的样本数量， 类别数量
        frame_dists = frame_sim
        # dists维度为 query样本数量， support类别数量，帧数，帧数
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)
        result = torch.zeros(dists.shape[0], dists.shape[1]).cuda()
        for i, item in enumerate(dists):
            for j, ittem in enumerate(dists[i]):
                sum_of_diagonal = torch.diag(dists[i,j,:,:]).sum()
                result[i][j] = sum_of_diagonal
        return result


    def otam_distance(self, support_features, target_features):
        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]
        support_features = rearrange(support_features, 'b s d -> (b s) d')  # 5 8 1024-->40  1024
        target_features = rearrange(target_features, 'b s d -> (b s) d')
        frame_sim = cos_sim(target_features, support_features)  # 类别数量*每个类的样本数量， 类别数量
        frame_dists = 1 - frame_sim
        # dists维度为 query样本数量， support类别数量，帧数，帧数
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)  # [25, 25, 8, 8]
        # calculate query -> support and support -> query  双向匹配还是单向匹配
        if self.argss.SINGLE_DIRECT:
            cum_dists = OTAM_cum_dist_v2(dists)
        else:
            cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
        return cum_dists

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())

    def distribute_model(self):

        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        gpus_use_number = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if self.argss.num_gpus > 1:
            self.backbone.cuda()
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(gpus_use_number)])
