import gc
import torch
from local_experiments import TestCMR

"""
def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('number', type=int, help='')
    args = parser.parse_args()

    

    if args.number == 1:
        pass
    elif args.number == 2:
        pass
    elif args.number == 3:
        pass
    elif args.number == 4:
        pass
    elif args.number == 5:
        pass
    elif args.number == 6:
        pass
"""


if __name__ == '__main__':

    gc.collect()
    torch.cuda.empty_cache()

    TestCMR().train_extended_cmr_mnist()
    TestCMR().test_rule_selector_rule_switch()
