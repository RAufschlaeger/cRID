# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if len(sys.argv) == 1:  # No arguments were provided
        args.config_file = './configs/market1501/dinov2_vits14_softmax_triplet_with_center.yml'
        args.opts = [
            'MODEL.DEVICE_ID', "('0')",  # Must be a string with quotes to match expected format
            'DATASETS.NAMES', "('market1501')",
            'DATASETS.ROOT_DIR', "('./data')",
            'OUTPUT_DIR', "('./logs/test/market1501/dinov2_vits14/softmax_triplet_with_center')"
        ]
        print("No arguments provided. Using default configuration:")
        print(f"  --config_file={args.config_file}")
        for i in range(0, len(args.opts), 2):
            print(f"  {args.opts[i]} {args.opts[i+1]}")

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    # Check if config file exists before attempting to load it
    if args.config_file != "":
        if not os.path.exists(args.config_file):
            print(f"Warning: Config file '{args.config_file}' does not exist. Creating directories and continuing...")
            # Create directory structure if it doesn't exist
            os.makedirs(os.path.dirname(args.config_file), exist_ok=True)
        else:
            cfg.merge_from_file(args.config_file)
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory is created

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    logger.info(f"Loading model checkpoint from {cfg.TEST.WEIGHT}")
    print(f"Model checkpoint path: {cfg.TEST.WEIGHT}")
    
    # Both .pth and .pt extensions are supported by PyTorch's loading mechanism
    if not os.path.exists(cfg.TEST.WEIGHT):
        logger.error(f"Checkpoint file not found: {cfg.TEST.WEIGHT}")
        print(f"ERROR: Checkpoint file not found: {cfg.TEST.WEIGHT}")
        sys.exit(1)
        
    # Load checkpoint and adapt format if needed
    try:
        checkpoint = torch.load(cfg.TEST.WEIGHT)
        logger.info(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'not a dict'}")
        
        # If the checkpoint doesn't have 'model' key but has state_dict directly
        if isinstance(checkpoint, dict) and 'model' not in checkpoint and next(iter(checkpoint.keys()), '').startswith('base'):
            logger.info("Converting checkpoint format - wrapping state_dict with 'model' key")
            adapted_checkpoint = {'model': checkpoint}
            # Create a temporary file with the adapted checkpoint
            temp_path = cfg.TEST.WEIGHT + '.adapted'
            torch.save(adapted_checkpoint, temp_path)
            model.load_param(temp_path)
            # Clean up temporary file
            os.remove(temp_path)
        else:
            # Try loading original checkpoint
            model.load_param(cfg.TEST.WEIGHT)
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        print(f"ERROR: Failed to load checkpoint: {str(e)}")
        sys.exit(1)

    inference(cfg, model, val_loader, num_query)


if __name__ == '__main__':
    main()
