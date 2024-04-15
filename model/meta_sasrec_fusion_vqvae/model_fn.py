import torch.nn
import torch.nn as nn

from module import encoder, common, initializer
from util import consts
from . import config
from ..model_meta import MetaType, model

import logging
logger = logging.getLogger(__name__)


@model("meta_sasrec_fusion_vqvae", MetaType.ModelBuilder)
class SasRec(nn.Module):
    def __init__(self, model_conf, pretrain_category_weight=None):
        super(SasRec, self).__init__()

        assert isinstance(model_conf, config.ModelConfig)

        self._item_position_embedding = encoder.IDEncoder(
            model_conf.item_id_vocab,
            model_conf.id_dimension
        )
                
        self._category_position_embedding = encoder.IDEncoder(
            model_conf.category_id_vocab,
            model_conf.id_dimension
        )

        self._item_id_encoder = encoder.IDEncoder(
            model_conf.item_id_vocab,
            model_conf.id_dimension
        )

        self._category_id_encoder = encoder.IDEncoder(
            model_conf.category_id_vocab,
            model_conf.id_dimension
        )

        self._item_target_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension] * 2, [torch.nn.Tanh, None]
        )
        self._category_target_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension] * 2, [torch.nn.Tanh, None]
        )
        self._item_seq_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )
        self._category_seq_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )

        self._item_transformer = nn.TransformerEncoderLayer(
            d_model=model_conf.id_dimension,
            nhead=model_conf.nhead,
            dim_feedforward=4*model_conf.id_dimension,
            dropout=0
        )
        self._category_transformer = nn.TransformerEncoderLayer(
            d_model=model_conf.id_dimension,
            nhead=model_conf.nhead,
            dim_feedforward=4*model_conf.id_dimension,
            dropout=0
        )

        initializer.default_weight_init(self._item_transformer.self_attn.in_proj_weight)
        initializer.default_weight_init(self._item_transformer.self_attn.out_proj.weight)
        initializer.default_bias_init(self._item_transformer.self_attn.in_proj_bias)
        initializer.default_bias_init(self._item_transformer.self_attn.out_proj.bias)

        initializer.default_weight_init(self._item_transformer.linear1.weight)
        initializer.default_bias_init(self._item_transformer.linear1.bias)
        initializer.default_weight_init(self._item_transformer.linear2.weight)
        initializer.default_bias_init(self._item_transformer.linear2.bias)

        initializer.default_weight_init(self._category_transformer.self_attn.in_proj_weight)
        initializer.default_weight_init(self._category_transformer.self_attn.out_proj.weight)
        initializer.default_bias_init(self._category_transformer.self_attn.in_proj_bias)
        initializer.default_bias_init(self._category_transformer.self_attn.out_proj.bias)

        initializer.default_weight_init(self._category_transformer.linear1.weight)
        initializer.default_bias_init(self._category_transformer.linear1.bias)
        initializer.default_weight_init(self._category_transformer.linear2.weight)
        initializer.default_bias_init(self._category_transformer.linear2.bias)

        self._meta_classifier_param_list = common.HyperNetwork_FC_Fusion_VQVAE(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None],
            batch=True,
            model_conf=model_conf,
            pretrain_category_weight=pretrain_category_weight
        )

    def forward(self, item_features, category_features, norm=0.01, fig1=False):
        # Encode target item
        item_trigger_embed = self._item_id_encoder(item_features[consts.FIELD_TRIGGER_SEQUENCE])
        item_trigger_embed = self._item_seq_trans(item_trigger_embed)

        # B * D
        item_target_embed = self._item_id_encoder(item_features[consts.FIELD_TARGET_ID])
        item_target_embed = self._item_target_trans(item_target_embed)

        # Encode user historical behaviors
        with torch.no_grad():
            item_click_seq = item_features[consts.FIELD_CLK_SEQUENCE]
            item_batch_size = int(item_click_seq.shape[0])
            # B * L
            item_positions = torch.arange(0, int(item_click_seq.shape[1]), dtype=torch.int32).to(item_click_seq.device)
            item_positions = torch.tile(item_positions.unsqueeze(0), [item_batch_size, 1])
            item_mask = torch.not_equal(item_click_seq, 0)
            # B
            item_seq_length = torch.maximum(torch.sum(item_mask.to(torch.int32), dim=1) - 1, torch.Tensor([0]).to(device=item_mask.device))
            item_seq_length = item_seq_length.to(torch.long)

            item_trigger_mask = torch.not_equal(item_features[consts.FIELD_TRIGGER_SEQUENCE], 0).to(dtype=torch.int32)
            # B
            item_trigger_seq_length = torch.maximum(torch.sum(item_trigger_mask, dim=1) - 1,
                                               torch.Tensor([0]).to(device=item_trigger_mask.device))
            item_trigger_seq_length = item_trigger_seq_length.to(torch.long)

        # B * L * D
        item_hist_embed = self._item_id_encoder(item_click_seq)
        item_hist_pos_embed = self._item_position_embedding(item_positions)
        item_hist_embed = self._item_seq_trans(item_hist_embed + item_hist_pos_embed)

        item_atten_embed = self._item_transformer(
            torch.swapaxes(item_hist_embed, 0, 1)
        )
        item_user_state = torch.swapaxes(item_atten_embed, 0, 1)[range(item_batch_size), item_seq_length, :]


        
        # Encode target category
        category_trigger_embed = self._category_id_encoder(category_features[consts.FIELD_TRIGGER_SEQUENCE])
        category_trigger_embed = self._category_seq_trans(category_trigger_embed)

        # B * D
        category_target_embed = self._category_id_encoder(category_features[consts.FIELD_TARGET_ID])
        category_target_embed = self._category_target_trans(category_target_embed)

        # Encode user historical behaviors
        with torch.no_grad():
            category_click_seq = category_features[consts.FIELD_CLK_SEQUENCE]
            category_batch_size = int(category_click_seq.shape[0])
            # B * L
            category_positions = torch.arange(0, int(category_click_seq.shape[1]), dtype=torch.int32).to(category_click_seq.device)
            category_positions = torch.tile(category_positions.unsqueeze(0), [category_batch_size, 1])
            category_mask = torch.not_equal(category_click_seq, 0)
            # B
            category_seq_length = torch.maximum(torch.sum(category_mask.to(torch.int32), dim=1) - 1, torch.Tensor([0]).to(device=category_mask.device))
            category_seq_length = category_seq_length.to(torch.long)

            category_trigger_mask = torch.not_equal(category_features[consts.FIELD_TRIGGER_SEQUENCE], 0).to(dtype=torch.int32)
            # B
            category_trigger_seq_length = torch.maximum(torch.sum(category_trigger_mask, dim=1) - 1,
                                               torch.Tensor([0]).to(device=category_trigger_mask.device))
            category_trigger_seq_length = category_trigger_seq_length.to(torch.long)

        # B * L * D
        category_hist_embed = self._category_id_encoder(category_click_seq)
        category_hist_pos_embed = self._category_position_embedding(category_positions)
        category_hist_embed = self._category_seq_trans(category_hist_embed + category_hist_pos_embed)

        category_atten_embed = self._category_transformer(
            torch.swapaxes(category_hist_embed, 0, 1)
        )
        category_user_state = torch.swapaxes(category_atten_embed, 0, 1)[range(category_batch_size), category_seq_length, :]



        user_embedding, vae_loss = self._meta_classifier_param_list(item_user_state, item_hist_embed,
                                                          category_user_state, category_hist_embed,
                                                          norm=norm,
                                                          sample_num=item_user_state.size()[0],
                                                          trigger_seq_length=item_seq_length)
        if fig1:
            user_embedding1 = self._meta_classifier_param_list(item_user_state, item_trigger_embed,
                                                               category_user_state, category_hist_embed,
                                                               norm=norm,
                                                               sample_num=item_user_state.size()[0],
                                                               trigger_seq_length=item_trigger_seq_length)
            output1 = torch.sum(user_embedding1 * item_target_embed, dim=1, keepdim=True)
            return output, mis_rec_pred, request_num, total_num, output1
        return torch.sum(user_embedding * item_target_embed, dim=1, keepdim=True), item_target_embed, vae_loss
