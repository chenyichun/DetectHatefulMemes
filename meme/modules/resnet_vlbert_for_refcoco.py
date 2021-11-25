import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBert, VisualLinguisticBertMVRCHeadTransform
import pdb

BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERT(Module):
    def __init__(self, config):

        super(ResNetVLBERT, self).__init__(config)

        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                enable_cnn_reg_loss=False)
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN
        self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)

        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path
        self.language_pretrained_model_path = language_pretrained_model_path
        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT,
                                         language_pretrained_model_path=language_pretrained_model_path)

        self.final_mlp_dims = config.NETWORK.FINAL_MLP_DIMS
        transform = VisualLinguisticBertMVRCHeadTransform(config.NETWORK.VLBERT)
        linear1 = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)
        ####### Set the output number of dimensions of final MLP with the given dims #######
        linear2 = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.FINAL_MLP_DIMS)
        self.final_mlp = nn.Sequential(
            transform,
            nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
            linear1,
            nn.ReLU(inplace=True),
            linear2
        )

        ####### Additional parameters to use in class methods #######
        self.weight_of_class_one = config.NETWORK.WEIGHT_OF_CLASS_ONE = 5200.0 / 2800.0
        self.pseudo_img_sa_weight = config.NETWORK.PSEUDO_IMG_SA_WEIGHT
        self.pseudo_img_sa_indices = config.NETWORK.PSEUDO_IMG_SA_INDICES
        self.pseudo_text_sa_weight = config.NETWORK.PSEUDO_TEXT_SA_WEIGHT
        self.pseudo_text_sa_indices = config.NETWORK.PSEUDO_TEXT_SA_INDICES
        self.pseudo_both_sa_weight = config.NETWORK.PSEUDO_BOTH_SA_WEIGHT
        self.pseudo_both_sa_indices = config.NETWORK.PSEUDO_BOTH_SA_INDICES

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(ResNetVLBERT, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass

    def train_forward(self,
                      image,
                      boxes,
                      im_info,
                      expression,
                      label
                      ):
        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        # origin_len = boxes.shape[1]
        ####### Is origin_len supposed to be 6 here? #######
        origin_len = self.final_mlp_dims # don't care number of boxes
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        # label = label[:, :max_len]
        label_img = label[:, 0:1]
        label_img_sa = label[:, 1:4]
        label_text_sa = label[:, 4:5]
        label_both_sa = label[:, 5:6]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states_text, hidden_states_regions, pooled_output = self.vlbert(text_input_ids,
                                                                 text_token_type_ids,
                                                                 text_visual_embeddings,
                                                                 text_mask,
                                                                 object_vl_embeddings,
                                                                 box_mask,
                                                                 output_all_encoded_layers=False,
                                                                 output_text_and_object_separately=True)

        ###########################################
        outputs = {}

        # classifier
        # logits = self.final_mlp(hidden_states_regions).squeeze(-1)
        logits = self.final_mlp(pooled_output)#.squeeze(-1)

        # loss
        loss = 0.0
        update_dict = dict()

        ####### Is class wieghts added this way? #######
        # cls_loss = F.binary_cross_entropy_with_logits(logits[box_mask], label[box_mask])
        # cls_loss = F.binary_cross_entropy_with_logits(logits[:, 0: 1], label_img.float(), pos_weight = torch.tensor([1.0, self.weight_of_class_one]).cuda())
        cls_loss = F.binary_cross_entropy_with_logits(logits[:, 0: 1], label_img.float(), pos_weight = torch.tensor([self.weight_of_class_one]).cuda())

        update_dict['cls_loss'] = cls_loss

        loss += cls_loss.mean()
        if self.pseudo_img_sa_weight:
            start_idx, end_idx = self.pseudo_img_sa_indices
            # print ('shape of pred, lab: ', logits[:, start_idx: end_idx].shape, label_img_sa.shape)
            # img_sa_loss = F.cross_entropy(logits[:, start_idx: end_idx], label_img_sa)
            img_sa_loss = self.pseudo_img_sa_weight * F.kl_div(logits[:, start_idx: end_idx], label_img_sa)
            loss += img_sa_loss
            update_dict['img_sa_loss'] = img_sa_loss
        if self.pseudo_text_sa_weight:
            start_idx, end_idx = self.pseudo_text_sa_indices
            text_sa_loss = self.pseudo_text_sa_weight * F.binary_cross_entropy_with_logits(logits[:, start_idx: end_idx], (label_text_sa + 1.0) / 2.0)
            loss += text_sa_loss
            update_dict['text_sa_loss'] = text_sa_loss
        if self.pseudo_both_sa_weight:
            start_idx, end_idx = self.pseudo_both_sa_indices
            both_sa_loss = self.pseudo_both_sa_weight * F.mse_loss(logits[:, start_idx: end_idx], label_both_sa)
            loss += both_sa_loss
            update_dict['both_sa_loss'] = both_sa_loss

        # pad back to origin len for compatibility with DataParallel
        logits_ = logits.new_zeros((logits.shape[0], origin_len)).fill_(-10000.0)
        logits_[:, :logits.shape[1]] = logits
        logits = logits_

        ####### Should it be detached? #######
        update_dict['label_logits'] = logits.detach()

        # label_ = label.new_zeros((logits.shape[0], origin_len)).fill_(-1)
        # label_[:, :label.shape[1]] = label
        # label = label_
        label_img_ = label_img.new_zeros((logits.shape[0], origin_len)).fill_(-1)
        label_img_[:, :label_img.shape[1]] = label_img
        label_img = label_img_

        update_dict['label'] = label_img.detach()


        ####### outputs is not modified according to multitasking losses yet #######
        # outputs.update({'label_logits': logits,
        #                 'label': label,
        #                 'cls_loss': cls_loss})

        outputs.update(update_dict)

        # loss = cls_loss.mean()

        return outputs, loss



    def inference_forward(self,
                          image,
                          boxes,
                          im_info,
                          expression):

        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        # origin_len = boxes.shape[1]
        origin_len = 1 # don't care number of boxes
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################
        # prepare text
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        text_input_ids = expression.new_zeros((expression.shape[0], expression.shape[1] + 2))
        text_input_ids[:, 0] = cls_id
        text_input_ids[:, 1:-1] = expression
        _sep_pos = (text_input_ids > 0).sum(1)
        _batch_inds = torch.arange(expression.shape[0], device=expression.device)
        text_input_ids[_batch_inds, _sep_pos] = sep_id
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = obj_reps['obj_reps'][:, 0].unsqueeze(1).repeat((1, text_input_ids.shape[1], 1))

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states_text, hidden_states_regions, pooled_output = self.vlbert(text_input_ids,
                                                                 text_token_type_ids,
                                                                 text_visual_embeddings,
                                                                 text_mask,
                                                                 object_vl_embeddings,
                                                                 box_mask,
                                                                 output_all_encoded_layers=False,
                                                                 output_text_and_object_separately=True)

        ###########################################
        outputs = {}

        # classifier
        # logits = self.final_mlp(hidden_states_regions).squeeze(-1)
        logits = self.final_mlp(pooled_output)#.squeeze(-1)
        logits = logits[:, 0].unsqueeze(1)

        # pad back to origin len for compatibility with DataParallel
        logits_ = logits.new_zeros((logits.shape[0], origin_len)).fill_(-10000.0)
        logits_[:, :logits.shape[1]] = logits
        logits = logits_

        w_ratio = im_info[:, 2]
        h_ratio = im_info[:, 3]
        pred_boxes = boxes[_batch_inds, logits.argmax(1), :4]
        pred_boxes[:, [0, 2]] /= w_ratio.unsqueeze(1)
        pred_boxes[:, [1, 3]] /= h_ratio.unsqueeze(1)
        outputs.update({'label_logits': logits,
                        'pred_boxes': pred_boxes})

        return outputs
