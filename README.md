# PyTorch 量化感知训练(Quantization Aware Training—QAT)

## 说明

- **训练数据：`cifar10`**

- **网络：mobilenet_v2和resnet18**

- **本项目只关注PyTorch 本身API的QAT(只能用CPU部署)的流程和速度提升，不关注具体test精度**

- **pytorch-quantization那套QAT请参考[pytorch-quantization’s documentation](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html)或[DEPLOYING QUANTIZATION AWARE TRAINED MODELS IN INT8 USING TORCH-TENSORRT](https://pytorch.org/TensorRT/_notebooks/vgg-qat.html)**


## 软件环境

- Ubuntu 20.04 x86_64

- python 3.9

- onnx==1.14.0

- onnxsim==0.4.33

- numpy==1.21.6

- torch==2.0.0+cu117

- torchvision==0.15.1+cu117

## 例子

- 安装

```shell
git clone https://github.com/jnulzl/PyTorch-QAT
cd https://github.com/jnulzl/PyTorch-QAT
pip install -r requirements.txt 
```

- MobileNetV2 QAT

```shell
python cifar.py #默认网络为mobilenet_v2
Files already downloaded and verified
Files already downloaded and verified
Training Model...
Epoch: 000 Eval Loss: 2.303 Eval Acc: 0.097
Epoch: 001 Train Loss: 2.024 Train Acc: 0.255 Eval Loss: 1.725 Eval Acc: 0.356
Epoch: 002 Train Loss: 1.693 Train Acc: 0.380 Eval Loss: 1.520 Eval Acc: 0.438
Epoch: 003 Train Loss: 1.533 Train Acc: 0.439 Eval Loss: 1.437 Eval Acc: 0.472
Epoch: 004 Train Loss: 1.441 Train Acc: 0.478 Eval Loss: 1.354 Eval Acc: 0.514
Epoch: 005 Train Loss: 1.368 Train Acc: 0.506 Eval Loss: 1.257 Eval Acc: 0.549
Epoch: 006 Train Loss: 1.289 Train Acc: 0.537 Eval Loss: 1.193 Eval Acc: 0.573
......
```

- ResNet18 QAT

更改`cifar10.py`中的`model_name = 'resnet18'`，然后
```shell
python cifar.py
```

最后训练并导出的`onnx`模型位于`save_models`目录

## 有用链接

- [官方文档quantization](https://pytorch.org/docs/stable/quantization.html)

- [官方教程STATIC QUANTIZATION WITH EAGER MODE IN PYTORCH](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

- [官方教程Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)

- [Pytorch实现量化感知训练QAT](https://manaai.cn/aicodes_detail3.html?id=73)

- [PyTorch Quantization Aware Training](https://github.com/leimao/PyTorch-Quantization-Aware-Training)

- [Pytorch 量化感知训练流程](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/AI%20%E9%83%A8%E7%BD%B2%E5%8F%8A%E5%85%B6%E5%AE%83%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/Pytorch%E9%87%8F%E5%8C%96%E6%84%9F%E7%9F%A5%E8%AE%AD%E7%BB%83%E8%AF%A6%E8%A7%A3/)

- [Pytorch实现卷积神经网络训练量化](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/AI%20%E9%83%A8%E7%BD%B2%E5%8F%8A%E5%85%B6%E5%AE%83%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/Pytorch%E5%AE%9E%E7%8E%B0%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AE%AD%E7%BB%83%E9%87%8F%E5%8C%96%EF%BC%88QAT%EF%BC%89/)

- [INT8 量化训练](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/AI%20%E9%83%A8%E7%BD%B2%E5%8F%8A%E5%85%B6%E5%AE%83%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/INT8%20%E9%87%8F%E5%8C%96%E8%AE%AD%E7%BB%83/)