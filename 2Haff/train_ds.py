import argparse
import cv2
import json
import os
import shutil
import sys
import time
import wandb
from functools import partial
from scipy.spatial.distance import directed_hausdorff


import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from model.segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPImageProcessor
import torch.nn.functional as F

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.aff_dataset import AffDataset, AffDatasetVal
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    # parser.add_argument(
    #     "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    # )
    parser.add_argument(
        "--version", default="liuhaotian/llava-v1.5-13b"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=575, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--reset_mask_decoder", action="store_true", default=False)
    parser.add_argument("--benchmark_dir", default='../../dataset/BENCHMARK/data/masks')
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    if args.reset_mask_decoder:
        for n, p in model.named_parameters():
            if "mask_decoder" in n:  # Check only for "mask_decoder"
                print("n: ", n, "p.shape: ", p.shape)
                
                # Reset weights (e.g., using Xavier initialization)
                if p.dim() > 1:  # For weights (not biases)
                    torch.nn.init.kaiming_uniform_(p, nonlinearity='relu')  # Or any other initialization method
                else:  # For biases
                    torch.nn.init.zeros_(p)

                p.requires_grad = True  # Ensure gradients are calculated for these parameters

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        False,
        samples_per_epoch=2
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        explanatory=args.explanatory,
    )

    if args.no_eval == False:
        """
        val_dataset = HybridDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            True,
            samples_per_epoch=args.batch_size
            * args.grad_accumulation_steps
            * args.steps_per_epoch
            * world_size,
            precision=args.precision,
            image_size=args.image_size,
            num_classes_per_sample=args.num_classes_per_sample,
            exclude_val=args.exclude_val,
            dataset=args.dataset,
            sample_rate=[float(x) for x in args.sample_rates.split(",")],
            sem_seg_data=args.sem_seg_data,
            refer_seg_data=args.refer_seg_data,
            vqa_data=args.vqa_data,
            reason_seg_data=args.reason_seg_data,
            explanatory=args.explanatory,
            #index_list = train_dataset.all_datasets[0].get_missing_indices()
        )
        """
        val_dataset = AffDatasetVal(
            args.benchmark_dir,
            tokenizer,
            args.vision_tower,
            args.precision,
            args.image_size,
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )
    
    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0
    """
    if args.eval_only:
        giou, ciou = validate(model_engine, 0, ,args)
        exit()
    """
    if args.local_rank == 0:
        wandb.init(
            project="2HAff_Bimanual",
            config={
                "dataset": "EPIC_Affordance",
            }
        )
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )
        
        if args.no_eval == False:
            iou, hd = validate(val_loader, model_engine, epoch, writer, args)
            is_best = iou > best_score
            best_score = max(iou, best_score)
            cur_ciou = iou if is_best else cur_ciou

        if args.no_eval or is_best:
            #if True:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):

    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    taxonomy_ce_losses = AverageMeter("TaxonomyCELoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)
            #import pdb; pdb.set_trace()
            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)
            # TODO: Add ce loss for taxonomy
            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            taxonomy_ce_loss = output_dict["taxonomy_ce_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            taxonomy_ce_losses.update(taxonomy_ce_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()
                taxonomy_ce_losses.all_reduce()

            if args.local_rank == 0:
                
                wandb.log({'train loss': losses.avg})
                wandb.log({'ce loss': ce_losses.avg})
                wandb.log({'mask bce loss': mask_bce_losses.avg})
                wandb.log({'mask dice loss': mask_dice_losses.avg})
                wandb.log({'taxonomy ce loss': taxonomy_ce_losses.avg})
                
                #print('train_loss: ',str(losses.avg))
                #print('ce_loss: ', str(ce_losses.avg))

                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()
            taxonomy_ce_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args):
    """
    intersection_meter_left = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    intersection_meter_right = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter_left = AverageMeter("Union", ":6.3f", Summary.SUM)
    union_meter_right = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    """
    model_engine.eval()
    ious = []
    hds = []
    iocms = []

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)
        # TODO: extend for two pred masks and mask the segmentations with taxonomy
        pred_masks_left = output_dict["pred_masks_left"]
        pred_masks_right = output_dict["pred_masks_right"]
        masks_list_left = output_dict["gt_masks_left"][0].int().cpu().numpy()
        masks_list_right = output_dict["gt_masks_right"][0].int().cpu().numpy()
        gt_aff = cv2.bitwise_or(masks_list_left, masks_list_right)
        #output_list_left = (pred_masks_left[0] > 0.5).int()
        #output_list_right = (pred_masks_right[0] > 0.5).int()
        pred_masks_left = (pred_masks_left[0] > 0).int()
        pred_masks_right = (pred_masks_right[0] > 0).int()
        assert len(pred_masks_left) == 1
        """
        intersection_left, intersection_right, union_left, union_right, acc_iou = 0.0, 0.0, 0.0, 0.0, 0.0
        for mask_i_left, mask_i_right, output_i_left, output_i_right in zip(masks_list_left, output_list_left, masks_list_right, output_list_right):
            intersection_i_left, union_i_left, _ = intersectionAndUnionGPU(
                output_i_left.contiguous().clone(), mask_i_left.contiguous(), 2, ignore_index=255
            )
            intersection_i_right, union_i_right, _ = intersectionAndUnionGPU(
                output_i_right.contiguous().clone(), mask_i_right.contiguous(), 2, ignore_index=255
            )
            intersection_left += intersection_i_left
            intersection_right += intersection_i_right
            union_left += union_i_left
            union_right += union_i_right
            acc_iou += (intersection_i_left / (union_i_left + 1e-5) + intersection_i_right / (union_i_right + 1e-5)) / 2
            # acc_iou[union_i == 0] += 1.0  # no-object target
        intersection_left, union_left = intersection_left.cpu().numpy(), union_left.cpu().numpy()
        intersection_right, union_right = intersection_right.cpu().numpy(), union_right.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list_left.shape[0]
        intersection_meter_left.update(intersection_left), union_meter_left.update(union_left), 
        intersection_meter_right.update(intersection_right), union_meter_right.update(union_right), 
        acc_iou_meter.update(acc_iou, n=masks_list_left.shape[0])
        """
        #import pdb; pdb.set_trace()
        taxonomy = output_dict["pred_taxonomies"][0]
        if taxonomy.numel() != 0:
            if torch.argmax(taxonomy) == 0:
                pred_mask = pred_masks_left.detach().cpu().numpy()[0]
                pred_mask[pred_mask > 0] = 255
                pred_mask[pred_mask <= 0] = 0

            elif torch.argmax(taxonomy) == 1:
                pred_mask = pred_masks_right.detach().cpu().numpy()[0]
                pred_mask[pred_mask > 0] = 255
                pred_mask[pred_mask <= 0] = 0
            else:
                pred_mask_left = pred_masks_left.detach().cpu().numpy()[0]
                pred_mask_right = pred_masks_right.detach().cpu().numpy()[0]
                pred_mask_left[pred_mask_left > 0] = 255
                pred_mask_left[pred_mask_left <= 0] = 0                    
                pred_mask_right[pred_mask_right> 0] = 255
                pred_mask_right[pred_mask_right <= 0] = 0
                pred_mask = cv2.bitwise_or(pred_mask_left, pred_mask_right)
        
        iou = calculate_iou(pred_mask, gt_aff)
        iocm = calculate_iocm(gt_aff, pred_mask)
        print(f"Precision: {iocm}")
        print(f"IoU: {iou}")
        #hd = calculate_hausdorff(pred_mask, gt_aff)
        ious.append(iou)
        iocms.append(iocm)
        #hds.append(hd)
        
    """
    intersection_meter_left.all_reduce()
    intersection_meter_right.all_reduce()
    union_meter_left.all_reduce()
    union_meter_right.all_reduce()
    acc_iou_meter.all_reduce()
    """
    """
    ciou = (intersection_meter_left.sum / (union_meter_left.sum + 1e-10) + intersection_meter_right.sum / (union_meter_right.sum + 1e-10)) / 2
    giou = acc_iou_meter.avg
    def compute_scalar_mean(data):
        # Check if the input is a PyTorch tensor
        if isinstance(data, torch.Tensor):
            # Compute mean and convert to a scalar if it's not 0-dimensional
            mean_value = data.mean()
            return mean_value.item() if mean_value.dim() == 0 else mean_value
        
        # Check if the input is a NumPy array
        elif isinstance(data, np.ndarray):
            # Compute mean and ensure it's a scalar
            mean_value = np.mean(data)
            return mean_value.item() if mean_value.ndim == 0 else mean_value
    giou = compute_scalar_mean(giou)
    ciou = compute_scalar_mean(ciou)
    if args.local_rank == 0:
        
        wandb.log({"giou": giou})
        wandb.log({"ciou": ciou})
        
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
    """
    total_iou = sum(ious)/len(ious)
    total_precision = sum(iocms)/len(iocms)
    #total_hd = sum(hds)/len(hds)
    print(total_iou)
    if args.local_rank == 0:
        wandb.log({"IoU": total_iou})
        wandb.log({"Precision": total_precision})
        #wandb.log({"HD": total_hd})
    return total_iou, 0

# def preprocess(
#     x,
#     pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
#     pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
#     img_size=1024,
# ) -> torch.Tensor:
#     """Normalize pixel values and pad to a square input."""
#     # Normalize colors
#     x = (x - pixel_mean) / pixel_std
#     # Pad
#     h, w = x.shape[-2:]
#     padh = img_size - h
#     padw = img_size - w
#     x = F.pad(x, (0, padw, 0, padh))
#     return x

def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    Args:mask1
    - mask1: First binary mask (numpy array).
    - mask2: Second binary mask (numpy array).
    Returns:
    - IoU value (float).
    """
    if np.array_equal(mask1, np.zeros((0, 0))) or np.array_equal(mask2, np.zeros((0, 0))):
        return

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union != 0 else 0.0
    return iou

def calculate_iocm(benchmark_mask, comparison_mask):
    """
    Calculate Intersection over Comparison Mask (IoCM) between two binary masks.
    
    Args:
    - benchmark_mask: The benchmark mask (numpy array).
    - comparison_mask: The comparison mask (numpy array).
    
    Returns:
    - IoCM value (float).
    """
    if np.array_equal(comparison_mask, np.zeros((0, 0))) or np.array_equal(benchmark_mask, np.zeros((0, 0))):
        return None  # If comparison mask is empty, return None
    
    intersection = np.logical_and(benchmark_mask, comparison_mask).sum()
    comparison_area = comparison_mask.sum()
    
    iocm = intersection / comparison_area if comparison_area != 0 else 0.0
    return iocm

def calculate_hausdorff(mask1, mask2):
    shp = mask1.shape
    mask1, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(mask2) == 0:
        print("I was here")
        return np.sqrt(shp[0]**2 + shp[1] ** 2), np.sqrt(shp[0]**2 + shp[1] ** 2)
    if len(mask1) == 0:
        return 0, 0
    mask1 = np.vstack(mask1[0]).squeeze()
    mask2 = np.vstack(mask2[0]).squeeze()
    if len(mask2.shape) == 1:
        mask2 = np.array([mask2])
    if len(mask1.shape) == 1:
        mask1 = np.array([mask1])
    return directed_hausdorff(mask2, mask1)[0], max(directed_hausdorff(mask1, mask2)[0], directed_hausdorff(mask2, mask1)[0])

# def validate(model_engine, epoch, tokenizer, clip_image_processor, args):
#     transform = ResizeLongestSide(args.image_size)
#     total_iou = 0
#     total_hd = 0
#     benchsize_ctr = 0
#     for folder_name in os.listdir(args.benchmark_dir):
#         subfolder_path = os.path.join(args.benchmark_dir, folder_name)
#         subfolders = os.listdir(subfolder_path)
#         for subfolder in subfolders:
#             folder_path = os.path.join(subfolder_path, subfolder)
            
#             if not os.path.isdir(folder_path):
#                 continue  # Skip non-directory files

#             image_path = os.path.join(folder_path, 'inpainting.png')
#             #image_path = os.path.join(folder_path, 'inpainting.png')
#             annotation_path = os.path.join(folder_path, 'annotation.json')
#             gt_aff_left_path = os.path.join(folder_path, 'aff_left.png')
#             gt_aff_right_path = os.path.join(folder_path, 'aff_right.png')
            
#             if not os.path.exists(image_path) or not os.path.exists(annotation_path):
#                 print(f"Required files not found in {folder_path}, skipping...")
#                 continue

#             if not os.path.exists(gt_aff_left_path):
#                 gt_aff = cv2.imread(gt_aff_right_path, cv2.IMREAD_GRAYSCALE)
#             elif not os.path.exists(gt_aff_right_path):
#                 gt_aff = cv2.imread(gt_aff_left_path, cv2.IMREAD_GRAYSCALE)
#             else:
#                 gt_aff_left = cv2.imread(gt_aff_left_path, cv2.IMREAD_GRAYSCALE)
#                 gt_aff_right = cv2.imread(gt_aff_right_path, cv2.IMREAD_GRAYSCALE)
#                 gt_aff = cv2.bitwise_or(gt_aff_left, gt_aff_right)

#             # Load the narration from the annotation.json
#             with open(annotation_path, 'r') as annotation_file:
#                 annotation_data = json.load(annotation_file)
#                 prompt = annotation_data.get('narration', '')

#             prompt = DEFAULT_IMAGE_TOKEN + "\n" + "Where would you interact with the object to perform action " + prompt
#             if args.use_mm_start_end:
#                 replace_token = (
#                     DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
#                 )
#                 prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            
#             # Load the image
#             image_np = cv2.imread(image_path)
#             image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#             original_size_list = [image_np.shape[:2]]

#             image_clip = (
#                 clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"]
#                 [0].unsqueeze(0).cuda()
#             )
#             if args.precision == "bf16":
#                 image_clip = image_clip.bfloat16()
#             elif args.precision == "fp16":
#                 image_clip = image_clip.half()
#             else:
#                 image_clip = image_clip.float()

#             image = transform.apply_image(image_np)
#             resize_list = [image.shape[:2]]

#             image = (
#                 preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
#                 .unsqueeze(0).cuda()
#             )
#             if args.precision == "bf16":
#                 image = image.bfloat16()
#             elif args.precision == "fp16":
#                 image = image.half()
#             else:
#                 image = image.float()

#             input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
#             input_ids = input_ids.unsqueeze(0).cuda()

#             output_ids, pred_masks_left, pred_masks_right, taxonomies = model_engine.evaluate(
#                 image_clip,
#                 image,
#                 input_ids,
#                 resize_list,
#                 original_size_list,
#                 max_new_tokens=512,
#                 tokenizer=tokenizer,
#             )
#             output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

#             #text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
#             #text_output = text_output.replace("\n", "").replace("  ", " ")
#             #print("text_output: ", text_output)
#             #import pdb; pdb.set_trace()
#             taxonomy = taxonomies[0]
#             if taxonomy.numel() != 0:
#                 if torch.argmax(taxonomy) == 0:
#                     pred_mask = pred_masks_left[0].detach().cpu().numpy()[0]
#                     pred_mask[pred_mask > 0] = 255
#                     pred_mask[pred_mask <= 0] = 0

#                 elif torch.argmax(taxonomy) == 1:
#                     pred_mask = pred_masks_right[0].detach().cpu().numpy()[0]
#                     pred_mask[pred_mask > 0] = 255
#                     pred_mask[pred_mask <= 0] = 0
#                 else:
#                     pred_mask_left = pred_masks_left[0].detach().cpu().numpy()[0]
#                     pred_mask_right = pred_masks_right[0].detach().cpu().numpy()[0]
#                     pred_mask_left[pred_mask_left > 0] = 255
#                     pred_mask_left[pred_mask_left <= 0] = 0                    
#                     pred_mask_right[pred_mask_right> 0] = 255
#                     pred_mask_right[pred_mask_right <= 0] = 0
#                     pred_mask = cv2.bitwise_or(pred_mask_left, pred_mask_right)
#             else:
#                 pred_mask_left = pred_masks_left[0].detach().cpu().numpy()[0]
#                 pred_mask_right = pred_masks_right[0].detach().cpu().numpy()[0]
#                 pred_mask_left[pred_mask_left > 0] = 255
#                 pred_mask_left[pred_mask_left <= 0] = 0                    
#                 pred_mask_right[pred_mask_right> 0] = 255
#                 pred_mask_right[pred_mask_right <= 0] = 0
#                 pred_mask = cv2.bitwise_or(pred_mask_left, pred_mask_right)
#             iou = calculate_iou(pred_mask, gt_aff)
#             hd = calculate_hausdorff(pred_mask, gt_aff)
#             total_hd += hd
#             total_iou += iou
#             benchsize_ctr += 1
#     wandb.log({'iou': total_iou/benchsize_ctr})
#     wandb.log({'hd': total_hd/benchsize_ctr})




if __name__ == "__main__":
    main(sys.argv[1:])
