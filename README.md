<h4 align="center"><strong><a href="https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence">Published: Engineering Applications of Artificial Intelligence (SCIE | Q1)</a></strong></h4>
<h2 align="center"><strong>Resource constraint crop damage classification using depth channel shuffling<a href="https://tanvirnwu.github.io/assets/papers/LightCDC.pdf" target="_blank">[Paper]</a></strong></h2>
<h6 align="center">Md Tanvir Islam<sup> 1</sup>, Safkat Shahrier Swapnil<sup> 2</sup>, Md. Masum Billal<sup> 3</sup>, Asif Karim<sup> 4, *</sup>, Niusha Shafiabady<sup> 5</sup>, Md. Mehedi Hassan<sup> 6, *</sup></h6>
<h6 align="center">| 1. Sungkyunkwan University, South Korea | 2. RUET, Bangladesh | 3. USTC, China | 4. Charles Darwin University, Australia | 5. Australian Catholic University, Australia | 6. Khulna University, Bangladesh || *Corresponding Authors |</h6> 
<hr>


## CDC Dataset & Model Weight
[Download Dataset (already available)](https://www.kaggle.com/datasets/tanvirnwu/crop-damage-classification-dataset-cdc-dataset) | Download Model Weight (will upload soon)


## Dependencies
#### Create conda environment
```
conda create --name lightCDC python=3.9
```
#### Install CUDA: Tested using CUDA 11.8
```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
```
#### Install other libraries
```
pip install -r requirements.txt
````

## Training

```
python train.py --config <name_of_your_config> --batch_size <batch_size> --lr <learning_rate>
```
For other paramaetrs you can edit the configs.py file of the config folder.

## Testing

#### Single Image Inference
```
python test.py --mode single --model_weight <model_weight_path> --image_path <your_image_path>
```

#### Multiple Image Inference (Folder)
```
python inference.py --mode multiple --model_weight <model_weight_path>
```

![Visitor Count](https://komarev.com/ghpvc/?username=tanvirnwu&repo=LightCDC_EAAI_2025)

## Cite this Paper

If you find our work useful in your research, please consider citing our paper and star ✨✨ this repository. Thank you!
```bibtex
@article{islam2025resource,
  title={Resource constraint crop damage classification using depth channel shuffling},
  author={Islam, Md Tanvir and Swapnil, Safkat Shahrier and Billal, Md Masum and Karim, Asif and Shafiabady, Niusha and Hassan, Md Mehedi},
  journal={Engineering Applications of Artificial Intelligence},
  volume={144},
  pages={110117},
  year={2025},
  publisher={Elsevier}
}

