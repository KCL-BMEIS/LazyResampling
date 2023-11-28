import os
import sys
import shutil
import argparse
from datetime import datetime
import json
import pprint

import sys


context = {
  "network": {
    "dimensions": 3,
    "image_channels": 4,
    "label_channels": 1,
    "seg_output_channels": 4,
    "img_output_channels": 1,
    "layer_count": 5,
    "layer_channels": [
      32,
      64,
      128,
      256,
      512
    ],
    "encoder_block_counts": [
      1,
      2,
      2,
      2,
      4
    ],
    "decoder_block_counts": [
      1,
      1,
      1,
      1,
      2
    ],
    "num_classes": 4,
    "network_type": "dynamic_guunet",
    "activation": "silu",
    "segmentation_loss": "dice",
    "autoencoder_loss": "xent",
    "latent_channels": 256,
    "var_clamp": 5,
    "var_denominator": 100,
    "var_weight_norm": False,
    "gaussian_sampler_softplus": False,
    "resolution_ratios": [
      [
        2,
        2,
        1
      ],
      [
        2,
        2,
        2
      ],
      [
        2,
        2,
        2
      ],
      [
        2,
        2,
        2
      ]
    ],
    "segmentation_mode": "basic",
    "autoencoder_mode": "off",
    "patch_size": [
      192,
      192,
      72
    ],
    "layer_dropouts": 0.05
  },
  "training": {
    "preprocessing": "training_lr_lazy_pipeline",
    "preprocessing_args": {
      "seed": 1493466461
    },
    "seg_factor": 1.0,
    "img_factor": 1.0,
    "epochs": 4096,
    "initializer_seed": 1493466461,
    "training_count": 64,
    "var_noise": 1.0,
    "telemetry_volumes_every": 1,
    "log_to_tensorboard": False,
    "timeout": 424800,
    "epoch_sample_count": 300,
    "can_save": True,
    "serialized_model": None,
    "logging_path": "/jmain02/home/J2AD019/exk01/mxm45-exk01/relight/logs/lr_01_unet_1_4_L192",
    "manifest": "/jmain02/home/J2AD019/exk01/mxm45-exk01/data/preprocessed/Task01_BrainTumour/orig/10_75_manifest_300_t_fold_0.json",
    "learning_rate_index": 44,
    "learning_rate": [
      [
        "scalar",
        6,
        1
      ],
      [
        "scalar",
        6,
        8
      ],
      [
        "scalar",
        3,
        4
      ],
      [
        "scalar",
        0,
        16
      ],
      [
        "stepped",
        1,
        16,
        1
      ]
    ],
    "batch_size": 2,
    "subsamples_per_sample": 1,
    "logging_tag": "unet/DS300_0_75_0/PS192_72_BS2/CH32/lazy_SLd_DR5_LRI44_LRS6_8_3_4_2_16_1_16_S1493466461"
  },
  "validation": {
    "batch_size": 1,
    "preprocessing": "validation_lr_lazy_pipeline",
    "preprocessing_args": {
      "norm": True
    },
    "validation_step": "simple",
    "validation_noise": "none",
    "log_to_tensorboard": False,
    "manifest": "/jmain02/home/J2AD019/exk01/mxm45-exk01/data/preprocessed/Task01_BrainTumour/orig/10_75_manifest_75_v_fold_0.json"
  }
}

network_context = context['network']
training_context = context['training']
validation_context = context['validation']


def run_setup(script_path):
    print(f"script_path:", script_path)

    job_path, job_name = os.path.dirname(script_path), os.path.basename(script_path)
    logging_path = training_context['logging_path']
    tag = training_context['logging_tag']
    flat_tag = tag.replace('/', '_')
    full_logging_path = os.path.dirname(os.path.join(logging_path, flat_tag))

    run_name = f'{flat_tag}_{datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}'
    # iterate until the suffix makes the name unique (note that this shouldn't
    # technically ever need to run more than once)
    suffix = ''
    i = 0
    while True:
        full_name = os.path.join(full_logging_path, run_name + suffix)
        if not os.path.exists(full_name):
            break
        i += 1
        suffix = '_{}'.format(i)

    print(f"full_name '{full_name}'")
    print("would run setup")
    os.makedirs(full_name)

    shutil.copy2(script_path, os.path.join(full_name, job_name))

    with open(os.path.join(full_name, 'network_context.json'), 'w') as f:
        json.dump(network_context, f, indent=2)

    with open(os.path.join(full_name, 'training_context.json'), 'w') as f:
        json.dump(training_context, f, indent=2)

    with open(os.path.join(full_name, 'validation_context.json'), 'w') as f:
        json.dump(validation_context, f, indent=2)


def run_execute(script_path):
    try:
        import relight
    except Exception as e:
    #    fixed_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        fixed_path = '/jmain02/home/J2AD019/exk01/mxm45-exk01/git/relight'
        print("Appending relight path '{}' to env".format(fixed_path))
        sys.path.append(fixed_path)
        import relight

    print("relight:", relight.__file__)

    from relight.engines.basic_training_and_validation import run_training

    print("would run training")
    print(script_path,
          network_context,
          training_context,
          validation_context)
    run_training(script_path, network_context, training_context, validation_context)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setup', action="store_true")
    args = parser.parse_args()

    script_path = os.path.abspath(__file__)

    if args.setup is True:
        run_setup(script_path)
    else:
        run_execute(script_path)
