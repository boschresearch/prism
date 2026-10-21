# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from VAD
#   (https://github.com/hustvl/VAD, version 36047b6)
# Copyright (c) VAD authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import shutil

# ---------------------------------------------
import sys

sys.path.append("")
import numpy as np
import argparse
import mmcv
import os
import copy
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed

# from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.VAD.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import json
import wandb

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from mmcv.image import tensor2imgs
from mmdet3d.models import Base3DDetector, Base3DSegmentor, SingleStageMono3DDetector

# Modified from mmdet3d.apis.single_gpu_test to add sampling of multiple ego trajectories during inference
def single_gpu_test(
    model,
    data_loader,
    show=False,
    out_dir=None,
    show_score_thr=0.3,
    sample_n_ego_trajs=1,
    dataset_size=1.0,
    save_slp_query_path=None,
    eval_posterior=False,
):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    total_samples = len(dataset)

    # number of samples to process (ceil to avoid empty if fraction is small)
    max_samples = int(total_samples * dataset_size)
    max_samples = max(1, max_samples)

    prog_bar = mmcv.ProgressBar(max_samples)

    for i, data in enumerate(data_loader):
        if i >= max_samples:  # stop once limit is reached
            break

        with torch.no_grad():
            result = model(
                return_loss=False,
                rescale=True,
                sample_n_ego_trajs=sample_n_ego_trajs,
                save_slp_query_path=save_slp_query_path,
                eval_posterior=eval_posterior,
                **data,
            )

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor, SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data, result, out_dir=out_dir)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data["img"][0], torch.Tensor):
                    img_tensor = data["img"][0]
                else:
                    img_tensor = data["img"][0].data[0]
                img_metas = data["img_metas"][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]["img_norm_cfg"])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta["img_shape"]
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta["ori_shape"][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta["ori_filename"])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show, result[i], show=show, out_file=out_file, score_thr=show_score_thr
                    )
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def parse_wandb_dict(s):
    if s == "None":
        return None

    result = {}
    for item in s.split(","):
        key, value = item.split("=")
        # Check if value contains semicolons (list)
        # tags are separated by semicolons, shoule be converted to list
        if ";" in value:
            result[key] = value.split(";")
        else:
            result[key] = value

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--json_dir", help="json parent dir name file")  # NOTE: json file parent folder name
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase" "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument("--gpu-collect", action="store_true", help="whether to use gpu to collect results.")
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic", action="store_true", help="whether to set deterministic options for CUDNN backend."
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)

    ### CUSTOM ADDITIONS ###
    parser.add_argument(
        "--save_log_dir", type=str, default="test", help='Root directory to save logs and results. Default is "test".'
    )
    parser.add_argument(
        "--wandb_init_cfg",
        type=str,
        default="None",
        help="W&B initialization config in key=value format. Example: project=myproj group=exp1",
    )
    # optional epoch that is being evaluated, used for logging purposes
    parser.add_argument(
        "--epoch_to_eval", type=int, default=-1, help="Epoch number to evaluate, used for logging purposes."
    )

    parser.add_argument(
        "--dataset_size",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use for testing. Default is 1.0 (full dataset).",
    )

    parser.add_argument(
        "--probabilistic_pipeline",
        action="store_true",
        help="IF to use probabilistic pipeline (with sampling from prior).",
        default=False,
    )

    parser.add_argument(
        "--sample_n_ego_trajs",
        type=int,
        default=1,
        help="Number of ego trajectories to sample during inference (when probabilistic_pipeline is True). Default is 1 (deterministic). \
            Evaluation still will be done on sigle trajectory. Extra trajectories are used for visualization purposes only.",
    )

    parser.add_argument(
        "--save_slp_feats",
        action="store_true",
        help="Whether to save SLP features and queries during testing.",
        default=False,
    )

    parser.add_argument(
        "--eval_posterior",
        action="store_true",
        help="Whether to use posterior for evaluation (only matters if probabilistic pipeline is used).",
        default=False,
    )

    ### END CUSTOM ADDITIONS ###

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def convert_to_json_serializable(obj):
    """Convert tensor objects to JSON serializable types."""
    if torch.is_tensor(obj):
        obj = obj.detach().cpu().tolist()
        return obj
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        # assert False
        model = MMDataParallel(model, device_ids=[0])
        if args.probabilistic_pipeline:
            outputs = single_gpu_test(
                model,
                data_loader,
                args.show,
                args.show_dir,
                sample_n_ego_trajs=args.sample_n_ego_trajs,
                dataset_size=args.dataset_size,
                save_slp_query_path=Path(args.save_log_dir) / "feats" if args.save_slp_feats else None,
                eval_posterior=args.eval_posterior,
            )
        else:
            outputs = single_gpu_test(
                model,
                data_loader,
                args.show,
                args.show_dir,
                dataset_size=args.dataset_size,
            )
    else:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    tmp = {}
    tmp["bbox_results"] = outputs
    outputs = tmp
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            # assert False
            if isinstance(outputs, list):
                mmcv.dump(outputs, args.out)
            else:
                mmcv.dump(outputs["bbox_results"], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options

        # change save folder: add subfolder with specific epoch and time
        kwargs["jsonfile_prefix"] = osp.join(args.save_log_dir)

        # kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
        #    '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs["bbox_results"], **kwargs)

        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            results_dict, formatted_metrics_dict = dataset.evaluate(outputs["bbox_results"], **eval_kwargs)
            formatted_metrics_dict = convert_to_json_serializable(formatted_metrics_dict)
            print(f"Perception results: {results_dict}")

        ### CUSTOM ADDITIONS ###

        # save config file and file with all launching arguments
        if True:
            config_save_dir = kwargs["jsonfile_prefix"]
            os.makedirs(config_save_dir, exist_ok=True)

            # Save the config file
            config_filename = args.config.split("/")[-1]
            config_save_path = osp.join(config_save_dir, config_filename)
            shutil.copy2(args.config, config_save_path)
            print(f"Config file saved to: {config_save_path}")

            # Save the launching arguments to a JSON file
            args_filename = "command_args.json"
            args_save_path = osp.join(config_save_dir, args_filename)
            with open(args_save_path, "w") as f:
                json.dump(args.__dict__, f, indent=4)  # Use json.dump for JSON format
            print(f"Command-line arguments saved to: {args_save_path}")

            # Save the formatted metrics dictionary to a JSON file
            metrics_filename = "formatted_metrics.json"
            metrics_save_path = osp.join(config_save_dir, metrics_filename)
            with open(metrics_save_path, "w") as f:
                json.dump(formatted_metrics_dict, f, indent=4)
            print(f"Formatted metrics saved to: {metrics_save_path}")

        # log to wandb
        if args.wandb_init_cfg != "None":
            print("Logging to W&B...")
            args.wandb_init_cfg = parse_wandb_dict(args.wandb_init_cfg)
            wandb.init(**args.wandb_init_cfg)
            formatted_metrics_dict["epoch"] = args.epoch_to_eval  # add epoch to metrics dict, used for sorting
            wandb.log(formatted_metrics_dict)
            wandb.finish()
            print("W&B logging completed.")
        else:
            print("W&B logging not initialized, skipping...")

        ### END CUSTOM ADDITIONS ###

        ###
        # # # NOTE: record to json
        # json_path = args.json_dir
        # if not os.path.exists(json_path):
        #     os.makedirs(json_path)

        # metric_all = []
        # for res in outputs['bbox_results']:
        #     for k in res['metric_results'].keys():
        #         if type(res['metric_results'][k]) is np.ndarray:
        #             res['metric_results'][k] = res['metric_results'][k].tolist()
        #     metric_all.append(res['metric_results'])

        # print('start saving to json done')
        # with open(json_path+'/metric_record.json', "w", encoding="utf-8") as f2:
        #     json.dump(metric_all, f2, indent=4)
        # print('save to json done')


if __name__ == "__main__":
    main()
