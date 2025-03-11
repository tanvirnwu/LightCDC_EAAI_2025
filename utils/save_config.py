from config import configs


def save_config(model_name):

    save_file_path = r'F:\Research\CDC5k\Storage\Save_Configs\\' + model_name + '_Configs.txt'

    # Read the contents of Config.py
    with open(configs.config_file_path, 'r') as file:
        config_contents = file.read()

    # Write the contents to a new text file
    with open(save_file_path, 'w') as file:
        file.write(config_contents)

    print(f'All adjustable Parameter values has been saved to {save_file_path}')

