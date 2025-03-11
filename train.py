import argparse
from torchinfo import summary
import engines
from config import configs
import utils
from model import lightCDC

def main():
    parser = argparse.ArgumentParser(description="Train LightCDC Model")

    parser.add_argument('--config', type=str, required=True,
                        help='Name of the training configuration (e.g., lightCDC_ES10_ADAM)')
    parser.add_argument('--model_name', type=str, default="lightCDC",
                        help='Optional model name override. If not set, uses "your_model_version" + config.')
    parser.add_argument('--batch_size', type=int, default=configs.batch_size,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=configs.lr if hasattr(configs, 'lr') else 1e-3,
                        help='Learning rate for training')
    parser.add_argument('--input_size', type=int, nargs=4, default=[32, 3, 256, 256],
                        help='Input size for torchinfo.summary (e.g., 32 3 256 256)')

    args = parser.parse_args()

    model = lightCDC()

    print("Model Architecture:")
    print(model)

    print("\nModel Summary:")
    summary(model, input_size=tuple(args.input_size),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20, row_settings=["var_names"])

    model_name = args.model_name if args.model_name else "your_model_version_" + args.config

    utils.save_config(model_name=model_name)
    engines.train_model(
        model_object=model,
        model_name=model_name,
        batch_size=args.batch_size,
        lr=args.lr
    )

if __name__ == '__main__':
    main()
