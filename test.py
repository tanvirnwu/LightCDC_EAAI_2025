from model import lightCDC
from torchinfo import summary
import engines
import utils



img = r"D:\Research\CDC5k\Data\adverseweather\ND\Fog\Hard\dense\NC2.jpg"
utils.single_image_inference(model=lightCDC(), model_name="ShuffleNetV2_Custom_CDC_ES10_ADAM_50_64_le-3", image_path = img)



# # === Perform single image inference ===
# img_path = r"F:\Research\CDC5k\Data\FakeGPT\cat- running.jpg"
# Utils.single_image_inference(model = Ensemble5(), model_name = 'Ensemble5_CIFAKE_ES_150_64_le-3', image_path = img_path)

# model_config = ['_5_32_le-3', '_10_32_le-3', '_20_32_le-3']
# num_evl = len(model_config)

# for i in range(num_evl):
#     config = model_config[i]
#     Utils.multiple_inference(model = AlexNet(), model_name = 'AlexNet'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = ConvNextLarge(), model_name = 'ConvNextLarge'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = DenseNet121(), model_name = 'DenseNet121'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = DenseNet161(), model_name = 'DenseNet161'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = DenseNet169(), model_name = 'DenseNet169'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = DenseNet201(), model_name = 'DenseNet201'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = EfficientNet_B0(), model_name = 'EfficientNet_B0'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = EfficientNetV2Large(), model_name = 'EfficientNetV2Large'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = GoogleNet(), model_name = 'GoogleNet'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = Inception_V3(), model_name = 'Inception_V3'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = MNasNet(), model_name = 'MNasNet'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = MobileNetV2(), model_name = 'MobileNetV2'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = MobileNetV3(), model_name = 'MobileNetV3'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = ResNet50(), model_name = 'ResNet50'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = ResNet101(), model_name = 'ResNet101'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = ResNet152(), model_name = 'ResNet152'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = ShuffleNetV2(), model_name = 'ShuffleNetV2'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = SqueezeNet1_1(), model_name = 'SqueezeNet1_1'+config, evl_csv_file_name = evl_csv_file_name)
#     Utils.multiple_inference(model = VGG16(), model_name = 'VGG16'+config, evl_csv_file_name = evl_csv_file_name)
#
#     if i != 1: # As we did not train these models for 10 epochs
#         Utils.multiple_inference(model = WideResNet50_2()  , model_name = 'WideResNet50_2'+config, evl_csv_file_name = evl_csv_file_name)
#         Utils.multiple_inference(model = WideResNet101_2(), model_name = 'WideResNet101_2'+config, evl_csv_file_name = evl_csv_file_name)
#
# #
# Utils.multiple_inference(model = ConvNextLarge(), model_name ='ConvNextLarge_ES_50_64_le-3', evl_csv_file_name = evl_csv_file_name1)
# Utils.multiple_inference(model = ResNet152(), model_name ='ResNet152_ES_50_64_le-3', evl_csv_file_name = evl_csv_file_name1)
# Utils.multiple_inference(model = DenseNet201(), model_name ='DenseNet201_ES_50_64_le-3', evl_csv_file_name = evl_csv_file_name1)

# Utils.multiple_inference(model = None, model_name ='Stacking1', evl_csv_file_name = evl_csv_file_name)
# Utils.multiple_inference(model = None, model_name ='Stacking2', evl_csv_file_name = evl_csv_file_name)
# Utils.multiple_inference(model = None, model_name ='Stacking3', evl_csv_file_name = evl_csv_file_name)
# Utils.multiple_inference(model = None, model_name ='Stacking4', evl_csv_file_name = evl_csv_file_name)
# Utils.multiple_inference(model = None, model_name ='Stacking5', evl_csv_file_name = evl_csv_file_name)

# Utils.multiple_inference(model = None, model_name ='Voting1', evl_csv_file_name = evl_csv_file_name)
# Utils.multiple_inference(model = None, model_name ='Voting2', evl_csv_file_name = evl_csv_file_name)
# Utils.multiple_inference(model = None, model_name ='Voting3', evl_csv_file_name = evl_csv_file_name)
# Utils.multiple_inference(model = None, model_name ='Voting4', evl_csv_file_name = evl_csv_file_name)
# Utils.multiple_inference(model = None, model_name ='Voting5', evl_csv_file_name = evl_csv_file_name)