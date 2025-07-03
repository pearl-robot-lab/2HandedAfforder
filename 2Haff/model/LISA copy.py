from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained, is_sam_pretrained=True)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder_left.train()
            self.visual_model.mask_decoder_right.train()
            for param in self.visual_model.mask_decoder_left.parameters():
                param.requires_grad = True
            for param in self.visual_model.mask_decoder_right.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')


        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list_left: List[torch.FloatTensor],
        masks_list_right: List[torch.FloatTensor],
        taxonomies_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks_left = []
        pred_masks_right = []
        pred_taxonomies = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )

            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)

            # TODO: Extend for left and right
            low_res_masks_left, iou_predictions_left, pred_taxonomy = self.model.visual_model.mask_decoder_left(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_taxonomies.append(pred_taxonomy)
            pred_mask_left = self.model.visual_model.postprocess_masks(
                low_res_masks_left,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks_left.append(pred_mask_left[:, 0])

            low_res_masks_right, iou_predictions_right = self.model.visual_model.mask_decoder_right(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask_right = self.model.visual_model.postprocess_masks(
                low_res_masks_right,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks_right.append(pred_mask_right[:, 0])


        model_output = output
        gt_masks_left = torch.stack(masks_list_left, dim=0)
        gt_masks_right = torch.stack(masks_list_right, dim=0)
        pred_masks_left = torch.stack(pred_masks_left, dim=0)
        pred_masks_right = torch.stack(pred_masks_right, dim=0)
        gt_taxonomies = taxonomies_list
        pred_taxonomies = torch.stack(pred_taxonomies)
        # TODO: add ce loss and extend for left right
        if inference:
            return {
                "pred_masks_left": pred_masks_left,
                "pred_masks_right": pred_masks_right,
                "pred_taxonomies": pred_taxonomies,
                "gt_masks_left": gt_masks_left,
                "gt_masks_right": gt_masks_right,
                "gt_taxonomies": gt_taxonomies,
            }

        output = model_output.logits
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight

        mask_bce_loss_left = 0
        mask_bce_loss_right = 0

        mask_dice_loss_left = 0
        mask_dice_loss_right = 0
        taxonomy_ce_loss = 0

        num_masks = 0

        weight_decoder_left = gt_taxonomies[:, 0]
        weight_decoder_right = gt_taxonomies[:, 1]
        weight_decoder_both = (gt_taxonomies[:, 2] + gt_taxonomies[:, 3])
        #print("SHAPES: ", weight_decoder_left.size(), weight_decoder_both.size(), pred_masks_left.size())

        pred_masks_left = (weight_decoder_left.view(-1, 1, 1, 1) + weight_decoder_both.view(-1, 1, 1, 1)) * pred_masks_left
        pred_masks_right = (weight_decoder_right.view(-1, 1, 1, 1) + weight_decoder_both.view(-1, 1, 1, 1)) * pred_masks_right

        for batch_idx in range(len(pred_masks_left)):
            gt_mask_left = gt_masks_left[batch_idx]
            gt_mask_right = gt_masks_right[batch_idx]
            gt_taxonomy = gt_taxonomies[batch_idx]
            pred_mask_left = pred_masks_left[batch_idx]
            pred_mask_right = pred_masks_right[batch_idx]
            pred_taxonomy = pred_taxonomies[batch_idx]
            #print("Pred Taxonomy Size: ", pred_taxonomy.size())
            #print("GT Taxonomy SIze: ", gt_taxonomy.size())
            # TODO: add ce_loss for taxonomy and mask the segmentation loss with taxonomy
            assert (
                gt_mask_left.shape[0] == pred_mask_left.shape[0]
            ), "gt_mask_left.shape: {}, pred_mask_left.shape: {}".format(
                gt_mask_left.shape, pred_mask_left.shape
            )
            #print("Max Sigmoid Pred Mask: ", torch.max(torch.sigmoid(pred_mask_left)))
            #print("Max Sigmoid GT Mask: ", torch.max(torch.sigmoid(gt_mask_left)))
            #print("Max Pred taxonomy: ", torch.max(pred_taxonomy))
            mask_bce_loss_left += (
                sigmoid_ce_loss(torch.sigmoid(pred_mask_left), gt_mask_left/255, num_masks=gt_mask_left.shape[0])
                * gt_mask_left.shape[0]
            )
            mask_dice_loss_left += (
                dice_loss(pred_mask_left, gt_mask_left/255, num_masks=gt_mask_left.shape[0])
                * gt_mask_left.shape[0]
            )

            mask_bce_loss_right += (
                sigmoid_ce_loss(torch.sigmoid(pred_mask_right), gt_mask_right/255, num_masks=gt_mask_right.shape[0])
                * gt_mask_right.shape[0]
            )
            mask_dice_loss_right += (
                dice_loss(pred_mask_right, gt_mask_right/255, num_masks=gt_mask_right.shape[0])
                * gt_mask_right.shape[0]
            )

            num_masks += gt_mask_left.shape[0]

            taxonomy_ce_loss += (
                self.ce_loss(pred_taxonomy, gt_taxonomy.unsqueeze(0).float())
            )

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss_left / (num_masks + 1e-8) + self.bce_loss_weight * mask_bce_loss_right / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss_left / (num_masks + 1e-8) + self.dice_loss_weight * mask_dice_loss_right / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss + taxonomy_ce_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "taxonomy_ce_loss": taxonomy_ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks_left = []
            pred_masks_right = []
            taxonomies = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks_left, iou_predictions_left, taxonomy = self.model.visual_model.mask_decoder_left(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask_left = self.model.visual_model.postprocess_masks(
                    low_res_masks_left,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks_left.append(pred_mask_left[:, 0])
                taxonomies.append(taxonomy)

                low_res_masks_right, iou_predictions_right = self.model.visual_model.mask_decoder_right(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask_right = self.model.visual_model.postprocess_masks(
                    low_res_masks_right,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks_right.append(pred_mask_right[:, 0])

        return output_ids, pred_masks_left, pred_masks_right, taxonomies
