import yaml
import os
import options
import utils.utils as utils


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from trainer import Trainer


##### ---- Exp dirs ---- #####
parser = options.get_args_parser()



##### ---- Load Configs ---- #####
p = parser.parse_args()
if p.config is not None:
    with open(p.config, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)
    parser.set_defaults(**default_arg)
args = parser.parse_args()
utils.init_seed(args.seed)
# make dirs
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


###### ---- Training & Testing ---- #####
trainer = Trainer(args)
trainer.train()



