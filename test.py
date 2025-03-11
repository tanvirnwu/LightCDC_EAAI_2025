import argparse
import utils


def main():
    parser = argparse.ArgumentParser(description="LightCDC Inference Script")

    parser.add_argument('--mode', type=str, choices=['single', 'multiple'], required=True,
                        help='Inference mode: "single" for one image or "multiple" for folder')
    parser.add_argument('--model_weight', type=str, required=True,
                        help='Path to the model weight file (.pth)')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the input image (required if mode is "single")')
    parser.add_argument('--test_folder', type=str, default=None,
                        help='Path to the test image folder (required if mode is "multiple")')

    args = parser.parse_args()

    if args.mode == 'single':
        if not args.image_path:
            raise ValueError("Please provide --image_path for single image inference.")
        utils.single_image_inference(model_weight=args.model_weight, image_path=args.image_path)

    elif args.mode == 'multiple':
        if not args.test_folder:
            raise ValueError("Please provide --test_folder for multiple image inference.")
        utils.multiple_inference(model_weight=args.model_weight, test_folder=args.test_folder)


if __name__ == '__main__':
    main()
