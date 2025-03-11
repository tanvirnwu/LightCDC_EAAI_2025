from Proposed_Models import *
from Pretrained_Models import *
from torchinfo import summary
import Engines


# print(ShuffleNetV2_Custom())
# summary(ShuffleNetV2_Custom(), input_size=(32, 3, 256, 256), col_names=["input_size", "output_size", "num_params", "trainable"], col_width=20, row_settings=["var_names"])
# Utils.detailed_model_summary(model = ShuffleNetV2_Custom(), input_size=(32, 3, 256, 256))
# print(ShuffleNetV2())
# ====== MODEL TRAINING ======

# H_Net_v1 = H_Net_v1().to(device)
# print(H_Net_v2)
# summary(H_Net_v2, input_size=(32, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"], col_width=20, row_settings=["var_names"])
# Utils.print_model_details(H_Net_v2)
#
# config = '_CDC_ES10_ADAM_50_8_le-3' #SA means Scheduler Activate # DA means data augmentation
# Utils.save_config(model_name = 'ShuffleNetV2_Custom' + config)
# Engines.train_model(model_object = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, batch_size=8, lr = 0.001)
# Utils.multiple_inference( model = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, evl_csv_file_name = evl_csv_file_name, batch_size=8, lr = 0.001)
#
# config = '_CDC_ES10_ADAM_50_16_le-3' #SA means Scheduler Activate # DA means data augmentation
# Utils.save_config(model_name = 'ShuffleNetV2_Custom' + config)
# Engines.train_model(model_object = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, batch_size=16, lr = 0.001)
# Utils.multiple_inference( model = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, evl_csv_file_name = evl_csv_file_name, batch_size=16, lr = 0.001)

config = '_CDC_ES10_Adagrad_50_64_le-3' #SA means Scheduler Activate # DA means data augmentation
Utils.save_config(model_name = 'ShuffleNetV2_Custom' + config)
Engines.train_model(model_object = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, batch_size=64, lr = 0.001, optimizer = optimizer1)
Utils.multiple_inference( model = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, evl_csv_file_name = evl_csv_file_name, batch_size=64, lr = 0.001)

config = '_CDC_ES10_AdamW_50_64_le-3' #SA means Scheduler Activate # DA means data augmentation
Utils.save_config(model_name = 'ShuffleNetV2_Custom' + config)
Engines.train_model(model_object = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, batch_size=64, lr = 0.001, optimizer = optimizer2)
Utils.multiple_inference( model = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, evl_csv_file_name = evl_csv_file_name, batch_size=64, lr = 0.001)

config = '_CDC_ES10_RMSprop50_64_le-3' #SA means Scheduler Activate # DA means data augmentation
Utils.save_config(model_name = 'ShuffleNetV2_Custom' + config)
Engines.train_model(model_object = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, batch_size=64, lr = 0.001, optimizer = optimizer3)
Utils.multiple_inference( model = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, evl_csv_file_name = evl_csv_file_name, batch_size=64, lr = 0.001)

config = '_CDC_ES10_Adadelta_64_le-3' #SA means Scheduler Activate # DA means data augmentation
Utils.save_config(model_name = 'ShuffleNetV2_Custom' + config)
Engines.train_model(model_object = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, batch_size=64, lr = 0.001, optimizer = optimizer4)
Utils.multiple_inference( model = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, evl_csv_file_name = evl_csv_file_name, batch_size=64, lr = 0.001)

config = '_CDC_ES10_Adamax_64_le-3' #SA means Scheduler Activate # DA means data augmentation
Utils.save_config(model_name = 'ShuffleNetV2_Custom' + config)
Engines.train_model(model_object = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, batch_size=64, lr = 0.001, optimizer = optimizer5)
Utils.multiple_inference( model = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, evl_csv_file_name = evl_csv_file_name, batch_size=64, lr = 0.001)

config = '_CDC_ES10_ASGD_64_le-3' #SA means Scheduler Activate # DA means data augmentation
Utils.save_config(model_name = 'ShuffleNetV2_Custom' + config)
Engines.train_model(model_object = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, batch_size=64, lr = 0.001, optimizer = optimizer6)
Utils.multiple_inference( model = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, evl_csv_file_name = evl_csv_file_name, batch_size=64, lr = 0.001)

config = '_CDC_ES10_Rprop_64_le-3' #SA means Scheduler Activate # DA means data augmentation
Utils.save_config(model_name = 'ShuffleNetV2_Custom' + config)
Engines.train_model(model_object = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, batch_size=64, lr = 0.001, optimizer = optimizer7)
Utils.multiple_inference( model = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, evl_csv_file_name = evl_csv_file_name, batch_size=64, lr = 0.001)

config = '_CDC_ES10_SparseAdam_64_le-3' #SA means Scheduler Activate # DA means data augmentation
Utils.save_config(model_name = 'ShuffleNetV2_Custom' + config)
Engines.train_model(model_object = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, batch_size=64, lr = 0.001, optimizer = optimizer8)
Utils.multiple_inference( model = ShuffleNetV2_Custom(), model_name = 'ShuffleNetV2_Custom' + config, evl_csv_file_name = evl_csv_file_name, batch_size=64, lr = 0.001)



ble7(), model_name = 'Ensemble7' + config, evl_csv_file_name = evl_csv_file_name)