import argparse
from argparse import Action

class DictAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        result = {}
        for kv in values.split(','):
            k, v = kv.split('=')
            result[k.strip()] = eval(v)
        setattr(namespace, self.dest, result)

def get_args_parser():
    parser = argparse.ArgumentParser(description='Flora Framework for Zero-shot Skeleton Action Recognition',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## config
    parser.add_argument('--config', default='configs/synse/ntu60_xsub_unseen5.yaml', help='path to the configuration file')
    parser.add_argument('--task-name', type=str, default='ntu60_xsub_seen55_unseen5', help='task name')
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')

    # shiftgcn
    parser.add_argument("--num-class", type=int, default=55, help="seen classes for training")
    parser.add_argument("--num-point", type=int, default=25, help="human joints")
    parser.add_argument("--num-person", type=int, default=2, help="the number of max person")
    parser.add_argument("--graph", type=str, default="graph.ntu_rgb_d.Graph", help="graph")
    parser.add_argument("--graph_args", action=DictAction, default=dict(), help="graph args")

    ## flow matching arch
    parser.add_argument("--sigma-min", type=float, default=1e-5, help="sigma min")
    parser.add_argument("--sigma-max", type=float, default=1.0, help="sigma max")
    parser.add_argument("--ske-dim", type=int, default=256, help="skeleton input channels")
    parser.add_argument("--latent-dim", type=int, default=768, help="latent representation dimension")
    parser.add_argument("--text-dim", type=int, default=1536, help="text input channels")
    parser.add_argument("--dit-layers", type=int, default=4, help="depth of DiT")
    parser.add_argument("--num-heads", type=int, default=4, help="DiT heads")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="DiT mlp ratio")
    parser.add_argument("--lambda-cfm", type=float, default=0.1, help="lambda for contrastive flow matching loss")
    parser.add_argument("--lambda-align", type=float, default=0.1, help="lambda for cross-vae alignment loss")

    ## checkpoint
    parser.add_argument("--shiftgcn-checkpoint-path", type=str, default=None, help='path to the pretrained ShiftGCN model')
    parser.add_argument("--clip-checkpoint-path", type=str, default=None, help='path to the pretrained CLIP model')

    ## optimization
    parser.add_argument('--lr', default=1e-4, type=float, help='max learning rate')
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')

    ## dataloader 
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--align-iter', default=1000, type=int, help='number of total iterations for cross-vae alignment')
    parser.add_argument('--train-batch-size', default=256, type=int, help='train batch size')
    parser.add_argument('--test-batch-size', default=256, type=int, help='test batch size')
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')

    # feeder
    parser.add_argument('--feeder', default='feeders.feeder_ntu.Feeder_FM', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=32, help='the number of worker for data loader')
    parser.add_argument('--use-features', action='store_true', help='whether to use pre-trained features')
    parser.add_argument('--data-path', type=str, default='/root/autodl-tmp/Neuron/data/ntu60/NTU60_CS.npz', help='path to the skeleton data or feature')
    parser.add_argument('--low-shot', action='store_true', help='low-shot training setting')
    parser.add_argument('--percentage', type=float, default=0.1, help='percentage of samples for low-shot training')

    # test
    parser.add_argument('--step-ratio', type=float, default=0.5, help='ratio of steps for testing')
    
    

    # semantics data path
    parser.add_argument('--semantic-path', type=str, default='semantics', help='path to the sadave semantics')
    parser.add_argument('--max-length', type=int, default=35, help='maximum length of the text description')
    parser.add_argument('--topk', type=int, default=10, help='top k for class diversity promotion')
    parser.add_argument('--mixed-weights',type=float, default=[0.1, 0.1],nargs='+',help='weights for mixed semantics (ZSL & GZSL)')


    # GZSL
    parser.add_argument('--calibration-factor', type=float, default=0.025, help='calibration factor for GZSL')

    parser.add_argument('--setting', type=str, default='ZSL', help='ZSL, GZSL or All')

    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output/', help='output directory')

    return parser