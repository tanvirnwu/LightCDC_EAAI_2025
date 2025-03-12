import argparse
import utils
from config import configs

def main():
    parser = argparse.ArgumentParser(description="LightCDC Inference Script")

    parser.add_argument('--mode', type=str, choices=['single', 'multiple'], required=True,
                        help='Inference mode: "single" for one image or "multiple" for folder')
    parser.add_argument('--model_weight', type=str, default='./pretrained_weights/lightCDC.pth',
                        help='Path to the model weight file (.pth)')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the input image (required if mode is "single")')
    parser.add_argument('--test_folder', type=str, default=configs.test_path,
                        help='Path to the test image folder (required if mode is "multiple")')

    args = parser.parse_args()

    if args.mode == 'single':
        if not args.image_path:
            raise ValueError("Please provide --image_path for single image inference.")
        utils.single_image_inference(model_weight=args.model_weight, image_path=args.image_path)

    elif args.mode == 'multiple':
        if not args.test_folder:
            raise ValueError("Please provide --test_folder for multiple image inference.")
        utils.multiple_inference(model_weight=args.model_weight, batch_size=2, test_folder=args.test_folder)


if __name__ == '__main__':
    main()
# python test.py --mode single --model_weight pretrained_weights/ShuffleNetV2_Custom_CDC_ES10_ADAM_50_64_le-3.pth --image_path D:\Research\LightCDC_EAAI_2025\data\test\non_damaged\0b772174269a4086bf1ccd5fd3b61f8745a023be.jpg
