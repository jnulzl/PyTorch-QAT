# PyTorch 量化感知训练(Quantization Aware Training—QAT)

## 说明

- **训练数据：`cifar10`**

- **网络：mobilenet_v2和resnet18**

- **本项目只关注PyTorch 本身API的QAT(只能用CPU部署)的流程、速度提升以及原始模型和QAT后模型的精度差别**

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
cd data
chmod a+x download_data.sh
./download_data.sh
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
Epoch: 095 Train Loss: 0.381 Train Acc: 0.863 Eval Loss: 0.619 Eval Acc: 0.799
Epoch: 096 Train Loss: 0.381 Train Acc: 0.863 Eval Loss: 0.616 Eval Acc: 0.803
Epoch: 097 Train Loss: 0.384 Train Acc: 0.861 Eval Loss: 0.619 Eval Acc: 0.799
Epoch: 098 Train Loss: 0.381 Train Acc: 0.863 Eval Loss: 0.616 Eval Acc: 0.802
Epoch: 099 Train Loss: 0.386 Train Acc: 0.860 Eval Loss: 0.622 Eval Acc: 0.800
Epoch: 100 Train Loss: 0.382 Train Acc: 0.861 Eval Loss: 0.619 Eval Acc: 0.800
Training QAT Model...
Epoch: 000 Eval Loss: 0.634 Eval Acc: 0.795
Epoch: 001 Train Loss: 0.407 Train Acc: 0.853 Eval Loss: 0.628 Eval Acc: 0.797
Epoch: 002 Train Loss: 0.407 Train Acc: 0.853 Eval Loss: 0.630 Eval Acc: 0.794
Epoch: 003 Train Loss: 0.406 Train Acc: 0.853 Eval Loss: 0.629 Eval Acc: 0.798
Epoch: 004 Train Loss: 0.404 Train Acc: 0.853 Eval Loss: 0.623 Eval Acc: 0.794
Epoch: 005 Train Loss: 0.406 Train Acc: 0.854 Eval Loss: 0.617 Eval Acc: 0.799
Epoch: 006 Train Loss: 0.399 Train Acc: 0.855 Eval Loss: 0.623 Eval Acc: 0.797
Epoch: 007 Train Loss: 0.401 Train Acc: 0.854 Eval Loss: 0.629 Eval Acc: 0.793
Epoch: 008 Train Loss: 0.392 Train Acc: 0.858 Eval Loss: 0.637 Eval Acc: 0.794
Epoch: 009 Train Loss: 0.397 Train Acc: 0.857 Eval Loss: 0.631 Eval Acc: 0.793
Epoch: 010 Train Loss: 0.394 Train Acc: 0.857 Eval Loss: 0.641 Eval Acc: 0.797
......
FP32 evaluation accuracy: 0.800
INT8 evaluation accuracy: 0.797
FP32 CPU Inference Latency: 2.32 ms / sample
FP32 CUDA Inference Latency: 3.22 ms / sample
INT8 CPU Inference Latency: 1.56 ms / sample
INT8 JIT CPU Inference Latency: 0.57 ms / sample
```

从以上结果看出，FP32和Int8测试精度相当(0.800 vs 0.797)，但Int8的速度是FP32的约4倍(2.32ms -> 0.57)，由此看出QAT后的效果还是很明显的。

以上测试数据因训练参数和硬盘环境不同而不同。

- ResNet18 QAT

更改`cifar10.py`中的`model_name = 'resnet18'`，然后
```shell
python cifar.py
......
```

最后训练并导出的`onnx`模型位于`save_models`目录！

## 有用链接

以下是一些本人参考的链接：

- [官方文档quantization](https://pytorch.org/docs/stable/quantization.html)
- [官方教程STATIC QUANTIZATION WITH EAGER MODE IN PYTORCH](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [官方教程Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)
- [torchvision/models/quantization/mobilenetv2.py](https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/mobilenetv2.py)
- [torchvision/models/quantization/resnet.py](https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/resnet.py)
- [Pytorch实现量化感知训练QAT](https://manaai.cn/aicodes_detail3.html?id=73)
- [PyTorch Quantization Aware Training](https://github.com/leimao/PyTorch-Quantization-Aware-Training)
- [Pytorch 量化感知训练流程](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/AI%20%E9%83%A8%E7%BD%B2%E5%8F%8A%E5%85%B6%E5%AE%83%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/Pytorch%E9%87%8F%E5%8C%96%E6%84%9F%E7%9F%A5%E8%AE%AD%E7%BB%83%E8%AF%A6%E8%A7%A3/)
- [Pytorch实现卷积神经网络训练量化](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/AI%20%E9%83%A8%E7%BD%B2%E5%8F%8A%E5%85%B6%E5%AE%83%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/Pytorch%E5%AE%9E%E7%8E%B0%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AE%AD%E7%BB%83%E9%87%8F%E5%8C%96%EF%BC%88QAT%EF%BC%89/)
- [INT8 量化训练](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/AI%20%E9%83%A8%E7%BD%B2%E5%8F%8A%E5%85%B6%E5%AE%83%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/INT8%20%E9%87%8F%E5%8C%96%E8%AE%AD%E7%BB%83/)