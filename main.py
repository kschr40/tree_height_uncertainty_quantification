import getpass
import os
import shutil
import socket
import sys
import tempfile
import warnings
from contextlib import contextmanager

import wandb

from runner import Runner
from utilities import GeneralUtility

warnings.filterwarnings('ignore')
from time import sleep
# print("Waiting for 1 hour to avoid preemption on GCP...")
# sleep(int(60*60*1))

debug = "--debug" in sys.argv
defaults = dict(
    # System
    seed=1,

    # Data
    dataset='ai4forest_random_s1_s2_12_12_256_256',
    years = None,
    time_mode = None,    # Must be either 'channel', 'rescale', or None. 
    collapse_months=False,  # If True, the months are collapsed into one channel
    single_month_scaling=True,
    # batch_size=6,
    batch_size=6,

    # Architecture
    arch='single_year_unet3d',  # Defaults to unet
    backbone='resnet50',  # Defaults to resnet50
    use_pretrained_model=False,

    # Optimization
    optim='AdamW',  # Defaults to AdamW
    loss_name='gaussian_nll',  # Defaults to shift_l1
    # loss_name='quantile',  # Defaults to shift_l1
    # loss_name='lognormal_nll',  # Defaults to shift_l1
    # n_iterations=20000,
    n_iterations=5,#,
    # log_freq=0.1, #1,
    log_freq=1, #1,

    initial_lr=1e-3,
    weight_decay=1e-2,

    # Efficiency
    fp16=False,
    num_workers_per_gpu=16,   # Defaults to 8
    prefetch_factor=None,

    # Other
    use_grad_clipping=True,
    use_weighted_sampler=None,  # Currently deactivated
    early_stopping=True,  # Flag for early stopping
    
    # Scaling adjustments
    scale_adjust_1234=-0.1,  # Adjustment for channels 1, 2, 3, 4.
    scale_adjust_6789=0.0,  # Adjustment for channels 6, 7, 8, 9.
    scale_adjust_0=0.0,     # Adjustment for channel 0.
    scale_adjust_51011=0.0, # Adjustment for channels 5, 10, 11.
    model_paths = {'initial': 'pretrained_models/unet3d_big_descent-sweep-2.pt'}
    #model_paths = {'initial': 'pretrained_models/quantile_regression_overnight.pt'}
)

# if not debug:
#     # Set everything to None recursively
#     defaults = GeneralUtility.fill_dict_with_none(defaults)

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity='ai4forest',  # automatically changed in sweep
)
config = wandb.config
config = GeneralUtility.update_config_with_default(config, defaults)


@contextmanager
def tempdir():
    username = getpass.getuser()
    tmp_root = '/scratch/local/' + username
    tmp_path = os.path.join(tmp_root, 'tmp')
    if os.path.isdir('/scratch/local/') and not os.path.isdir(tmp_root):
        os.mkdir(tmp_root)
    if os.path.isdir(tmp_root):
        if not os.path.isdir(tmp_path): os.mkdir(tmp_path)
        path = tempfile.mkdtemp(dir=tmp_path)
    else:
        assert 'htc-' not in os.uname().nodename, "Not allowed to write to /tmp on htc- machines."
        path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write('Failed to clean up temp dir {}'.format(path))


with tempdir() as tmp_dir:
    # Check if we are running on the GCP cluster, if so, mark as potentially preempted
    is_htc = 'htc-' in os.uname().nodename
    is_gcp = 'gpu' in os.uname().nodename and not is_htc
    if is_gcp:
        print('Running on GCP, marking as preemptable.')
        wandb.mark_preempting()  # Note: This potentially overwrites the config when a run is resumed -> problems with tmp_dir
    sys.stdout.write(str(config))
    runner = Runner(config=config, tmp_dir=tmp_dir, debug=debug)
    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
