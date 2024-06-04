# YoloV8 TensorRT Jetson (Orin) Nano
![TensorRTparking](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano/assets/44409029/7404518b-b9a3-49f2-945f-dd0839d18de3)
## YoloV8 with the TensorRT framework. <br/>
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)<br/><br/>
A lightweight C++ implementation of YoloV8 running on NVIDIAs TensorRT engine.<br>
No additional libraries are required, just a few lines of code using software, found on every JetPack.<br><br>
For now: https://github.com/akashAD98/yolov8_in_depth<br>
Paper: on Ultralytics TODO list https://github.com/ultralytics/ultralytics<br/><br/>
Specially made for a Jetson Nano see [Q-engineering deep learning examples](https://qengineering.eu/deep-learning-examples-on-raspberry-32-64-os.html)

![](./models/example.svg)


------------

## Model performance benchmark(FPS)

All models are quantized to `FP16`.<br>
The `int8` models don't give any increase in FPS, while, at the same time, their mAP is significantly worse.<br>
The numbers reflect only the inference timing. Grabbing frames, post-processing and drawing are not taken into account.<br>

| demo             | model_name                   | Orin Nano | Nano  |
| ---------------- | ---------------------------- | :-----: | :--------: |
| yolov5           | yolov5nu                     | 100     | 20         |
| yolov8           | yolov8n                      | 100     | 19         |
|                  | yolov8s                      | 100     | 9.25       |
|                  | yolov8m                      | 40      |  -        |
|                  | yolov8l                      | 20      |  -        |
|                  | yolov8x                      | 17      |  -        |

------------

## Dependencies.
To run the application, you have to:
- OpenCV 64-bit installed.
- Optional: Code::Blocks. (```$ sudo apt-get install codeblocks```)

### Installing the dependencies.
Start with the usual 
```
$ sudo apt-get update 
$ sudo apt-get upgrade
$ sudo apt-get install cmake wget curl
```
#### OpenCV
Follow the [Jetson Nano guide](https://qengineering.eu/install-opencv-on-jetson-nano.html) or the [Jetson Orin Nano guide](https://qengineering.eu/install-opencv-on-orin-nano.html). <br>

------------

## Installing the app.
To extract and run the network in Code::Blocks <br/>
```
$ mkdir *MyDir*
$ cd *MyDir* 
$ git clone --depth=1 https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano.git
```

------------

## Getting your onnx model.
You always start with an `onnx` YoloV8.2 model generated by [ultralytics](https://www.ultralytics.com/).<br>
There are three ways to obtain a model:<br>
- Use an `onnx` model from the `./models` folder.<br>
  * YoloV5nu.onnx
  * YoloV8n.onnx
  * YoloV8s.onnx
- Download an `onnx` model from our Sync drive.<br>
  * [YoloV8n.onnx](https://ln5.sync.com/dl/02a6b0730/isn4d4cw-4bfnu7p6-92nmtn7f-kn93hqs3)
  * [YoloV8s.onnx](https://ln5.sync.com/dl/0f3d84a10/i844hrtj-qavzg39e-d53u9zxr-d5qkbsz9)
  * [YoloV8m.onnx](https://ln5.sync.com/dl/ff04119e0/6q6we8ti-3x6i9hky-97jy3z76-g9invy68)
  * [YoloV8l.onnx](https://ln5.sync.com/dl/08b547ff0/5z8edbvq-vz6rwp5g-scj9hpwc-2xejss38)
  * [YoloV8x.onnx](https://ln5.sync.com/dl/f53d9b900/758dckst-wbvwzt8i-298mmy58-ujys8jwk)
- Export your (custom-trained) model by ultralytics.<br>
  * [Install ultralytics](https://docs.ultralytics.com/guides/nvidia-jetson/#install-ultralytics-package) with all its third-party software on your Jetson.
  * `$ export "PATH=$PATH:~/.local/bin/" >> ~/.bashrc`
  * `$ source ~/.bashrc`
  * `$ yolo export model=yolov8s.pt format=onnx opset=11 simplify=True`

------------

## Getting your engine model.
TensorRT works with `*.engine` models.<br>
The models must be generated by the **_same version_** as the TensorRT version on your Jetson, otherwise you run into errors.<br>
![TensorRTerror](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano/assets/44409029/ea874e50-3902-4127-86a0-9639ebdd63bc)
<br>
That's why we provide the underlying `onnx` models instead of the `engine` models.<br> 
You need your `trtexec` app on your Jetson to convert the model from onnx to the engine format.<br>
Usually, `trtexec` is found in the `/usr/src/tensorrt/bin/` folder on your Jetson.<br>
You could include this folder in your PATH with the next command.<br>
```script
$ export "PATH=$PATH:/usr/src/tensorrt/bin/" >> ~/.bashrc
$ source ~/.bashrc
```
To export `onnx` to `engine` use the following command.<br>
```
$ trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16
```
Please be patient, it will take minutes to complete the conversion.<br><br>
Instead of `--fp16`, you could use `--int8`. All 16-bit floating points are now pruned to 8-bit integers, giving you a smaller but less accurate model. You can run the app once you have your `yolov8s.engine` model.

------------

## Running the app.
You can use **Code::Blocks**.
- Load the project file *.cbp in Code::Blocks.
- Select _Release_, not Debug.
- Compile and run with F9.
- You can alter command line arguments with _Project -> Set programs arguments..._ 

Or use **Cmake**.
```
$ cd *MyDir*
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
```

------------

### Thanks.
A more than special thanks to [***triple-Mu***](https://github.com/triple-Mu).<br><br>
![TensorRTbusstop](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano/assets/44409029/6a397012-9541-492a-8867-c24d06ebfbad)<br>

------------

[![paypal](https://qengineering.eu/images/TipJarSmall4.png)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=CPZTM5BB3FCYL) 
