# pix2code
*Generating Code from a Graphical User Interface Screenshot*   从图形用户界面截图生成代码

[![License](http://img.shields.io/badge/license-APACHE2-blue.svg)](LICENSE.txt)

* A video demo of the system can be seen 该系统的视频演示 [here](https://youtu.be/pqKeXkhFA3I)
* The paper is available at [https://arxiv.org/abs/1705.07962](https://arxiv.org/abs/1705.07962)
* Official research page 官方网页: [https://uizard.io/research#pix2code](https://uizard.io/research#pix2code)

## Abstract 摘要
Transforming a graphical user interface screenshot created by a designer into computer code is a typical task conducted by a developer in order to build customized software, websites, and mobile applications. In this paper, we show that deep learning methods can be leveraged to train a model end-to-end to automatically generate code from a single input image with over 77% of accuracy for three different platforms (i.e. iOS, Android and web-based technologies).
将设计人员设计的图形用户界面截屏转换为计算机代码是开发人员为实现定制的软件、网站和移动应用程序而执行的典型任务。在本文中，我们可以利用深度学习的方法来训练一个端到端模型，以便在三个不同的平台(iOS、Android和基于网络的技术)中，准确率超过77%地，从单输入图像自动生成代码。

## Citation 引文

```
@article{beltramelli2017pix2code,
  title={pix2code: Generating Code from a Graphical User Interface Screenshot},
  author={Beltramelli, Tony},
  journal={arXiv preprint arXiv:1705.07962},
  year={2017}
}
```

## Disclaimer 免责声明

The following software is shared for educational purposes only. The author and its affiliated institution are not responsible in any manner whatsoever for any damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of the use or inability to use this software.
以下软件仅用于教育目的。作者及其附属机构对任何损害不负任何责任，包括因使用或不能使用本软件而造成的任何直接、间接、特殊、附带或间接损害。

The project pix2code is a research project demonstrating an application of deep neural networks to generate code from visual inputs.
The current implementation is not, in any way, intended, nor able to generate code in a real-world context.
We could not emphasize enough that this project is experimental and shared for educational purposes only.
Both the source code and the datasets are provided to foster future research in machine intelligence and are not designed for end users.
pix2code项目展示了深层神经网络在将视觉输入转换为生成代码方面的应用。当前并没有这方面技术的实现，也没能够在现实情形中生成代码。这个项目是仅仅是试验性的，并且仅仅是为了教育目的而分享。源代码和数据集都是为了促进机器智能的未来研究而提供的，而不是为最终用户设计的。

## Setup 设置
### Prerequisites 先决条件

- Python 2 or 3
- pip

### Install dependencies 安装依赖性

```sh
pip install -r  requirements.txt
```

## Usage 使用

Prepare the data: 数据准备
```sh
# reassemble and unzip the data 重新组装并解压缩数据
cd datasets
zip -F pix2code_datasets.zip --out datasets.zip
unzip datasets.zip

cd ../model

# split training set and evaluation set while ensuring no training example in the evaluation set 确保评估集中没有训练集时分离训练集评估集
# usage: build_datasets.py <input path> <distribution (default: 6)>
./build_datasets.py ../datasets/ios/all_data
./build_datasets.py ../datasets/android/all_data
./build_datasets.py ../datasets/web/all_data

# transform images (normalized pixel values and resized pictures) in training dataset to numpy arrays (smaller files if you need to upload the set to train your model in the cloud) 将训练数据集中的图像(标准化像素值和调整大小的图片)转换为NumPy数组(如果需要上传集合以在云中训练模型，则减小文件)
# usage: convert_imgs_to_arrays.py <input path> <output path>
./convert_imgs_to_arrays.py ../datasets/ios/training_set ../datasets/ios/training_features
./convert_imgs_to_arrays.py ../datasets/android/training_set ../datasets/android/training_features
./convert_imgs_to_arrays.py ../datasets/web/training_set ../datasets/web/training_features
```

Train the model:
```sh
mkdir bin
cd model

# provide input path to training data and output path to save trained model and metadata 提供训练数据的输入路径和输出路径，来保存训练过的模型和元数据
# usage: train.py <input path> <output path> <is memory intensive (default: 0)> <pretrained weights (optional)>
./train.py ../datasets/web/training_set ../bin

# train on images pre-processed as arrays 对图像预处理为数组的训练
./train.py ../datasets/web/training_features ../bin

# train with generator to avoid having to fit all the data in memory (RECOMMENDED) 使用生成器进行训练，来避免必须适合内存中的所有数据
./train.py ../datasets/web/training_features ../bin 1

# train on top of pretrained weights 在预训练权重的基础上训练
./train.py ../datasets/web/training_features ../bin 1 ../bin/pix2code.h5
```

Generate code for batch of GUIs 多个GUI生成代码:
```sh
mkdir code
cd model

# generate DSL code (.gui file), the default search method is greedy 生成深度优先代码（.gui 文件），默认的搜索方法是贪心算法
# usage: generate.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>
./generate.py ../bin pix2code ../gui_screenshots ../code

# equivalent to command above 与上述命令等价
./generate.py ../bin pix2code ../gui_screenshots ../code greedy

# generate DSL code with beam search and a beam width of size 3 集束搜索并且集束宽度为3，生成DSL代码
./generate.py ../bin pix2code ../gui_screenshots ../code 3
```

Generate code for a single GUI image 单个GUI图像生成代码:
```sh
mkdir code
cd model

# generate DSL code (.gui file), the default search method is greedy 生成深度优先代码（.gui 文件），默认的搜索方法是贪心算法
# usage: sample.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>
./sample.py ../bin pix2code ../test_gui.png ../code

# equivalent to command above 与上述命令等价
./sample.py ../bin pix2code ../test_gui.png ../code greedy

# generate DSL code with beam search and a beam width of size 3 集束搜索并且集束宽度为3，生成DSL代码
./sample.py ../bin pix2code ../test_gui.png ../code 3
```

Compile generated code to target language 将生成的代码编译成目标语言:
```sh
cd compiler

# compile .gui file to Android XML UI
./android-compiler.py <input file path>.gui

# compile .gui file to iOS Storyboard
./ios-compiler.py <input file path>.gui

# compile .gui file to HTML/CSS (Bootstrap style)
./web-compiler.py <input file path>.gui
```

## FAQ 常见问题

### Will pix2code supports other target platforms/languages? pix2code是否支持其他目标平台/语言？
No, pix2code is only a research project and will stay in the state described in the paper for consistency reasons.
This project is really just a toy example but you are of course more than welcome to fork the repo and experiment yourself with other target platforms/languages.
不，出于一系列的原因，pix2code只是一个研究项目，并将停留在本文描述的状态。这个项目确实只是一个例子，但是非常欢迎你存储到自己的仓库，并且自己用其他的目标平台/语言去实现。

### Will I be able to use pix2code for my own frontend projects? 我是否能够在我自己的前端项目中使用pix2code？
No, pix2code is experimental and won't work for your specific use cases. 不，pix2code是实验性的，不适用于特定的用例。

### How is the model performance measured? 如何衡量模型的性能？
The accuracy/error reported in the paper is measured at the DSL level by comparing each generated token with each expected token.
Any difference in length between the generated token sequence and the expected token sequence is also counted as error.
本文的正确性/误差通过比较每个产生的标记和每个预期的标记，在DSL级别上进行测量。生成的标记序列与预期的标记序列之间的任何长度差异也被算为错误。

### How long does it take to train the model? 训练模型需要多长时间？
On a Nvidia Tesla K80 GPU, it takes a little less than 5 hours to optimize the 109 * 10^6 parameters for one dataset; so expect around 15 hours if you want to train the model for the three target platforms.
在Nvidia Tesla K80 GPU（高性能计算CPU）上，为一个数据集优化109*10^6个参数需要花费近5小时的时间；因此，如果您想要为三个目标平台训练模型，则需要15小时左右的时间。

### I am a front-end developer, will I soon lose my job? 我是一个前端开发商，我会很快失去我的工作吗？
*(I have genuinely been asked this question multiple times)* (我确实曾多次被问过这个问题)

**TL;DR** Not anytime soon will AI replace front-end developers. 人工智能不会在短期内取代前端开发者。

Even assuming a mature version of pix2code able to generate GUI code with 100% accuracy for every platforms/languages in the universe, front-enders will still be needed to implement the logic, the interactive parts, the advanced graphics and animations, and all the features users love. The product we are building at [Uizard Technologies](https://uizard.io) is intended to bridge the gap between UI/UX designers and front-end developers, not replace any of them. We want to rethink the traditional workflow that too often results in more frustration than innovation. We want designers to be as creative as possible to better serve end users, and developers to dedicate their time programming the core functionality and forget about repetitive tasks such as UI implementation. We believe in a future where AI collaborate with humans, not replace humans.
即使假设pix2code的成熟版本能够为世界上的每一种平台/语言生成100%正确的GUI代码，仍然需要前端来实现逻辑、交互部分、高级图形和动画以及用户喜欢的所有功能。我们正在 [Uizard Technologies](https://uizard.io) 生产的产品，目的是弥补UI/UX设计人员和前端开发人员之间的间断，而不是取代他们中的任何一个。我们想要重新思考传统的工作流程，这种工作流程往往会带来更多的挫折而非创新。我们希望设计人员尽可能具有创造性，以便更好地为终端用户服务，而开发人员则希望他们将时间用于核心功能的编程，并忘记重复的任务，如UI实现。我们相信，未来人工智能将与人类合作，而不是取代人类。

## Media coverage 媒体报道

* [Wired UK](http://www.wired.co.uk/article/pix2code-ulzard-technologies)
* [The Next Web](https://thenextweb.com/apps/2017/05/26/ai-raw-design-turn-source-code)
* [Fast Company](https://www.fastcodesign.com/90127911/this-startup-uses-machine-learning-to-turn-ui-designs-into-raw-code)
* [NVIDIA Developer News](https://news.developer.nvidia.com/ai-turns-ui-designs-into-code)
* [Lifehacker Australia](https://www.lifehacker.com.au/2017/05/generating-user-interface-code-from-images-using-machine-learning/)
* [Two Minute Papers](https://www.youtube.com/watch?v=Fevg4aowNyc) (web series)
* [NLP Highlights](https://soundcloud.com/nlp-highlights/17a) (podcast)
* [Data Skeptic](https://dataskeptic.com/blog/episodes/2017/pix2code) (podcast)
* Read comments on [Hacker News](https://news.ycombinator.com/item?id=14416530)
