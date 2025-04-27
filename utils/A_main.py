from models import get_all_models, get_model_class
from datasets import get_dataset_names, get_dataset_class
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False, add_help=False)
    parser.add_argument('--model', type=str, help='Model name.', choices=list(get_all_models().keys()))
    parser.add_argument('--dataset', type=str, required=True, choices=get_dataset_names())
    parser.add_argument('--buffer_size', type=int, required=True)
    parser.add_argument('--alpha', type=float, help='Penalty weight for MSE loss in buffer.')
    parser.add_argument('--beta', type=float, help='Penalty weight for CE loss in buffer.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dataset = get_dataset(args)
    model_class = get_model_class(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    transform = dataset.get_transform()
    model = model_class(backbone, loss, args, transform)
    train(model, dataset, args)

if __name__ == '__main__':
    main()
