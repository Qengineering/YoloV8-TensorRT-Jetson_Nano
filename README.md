# YoloV8 TensorRT Jetson (Orin) Nano
![output image]( https://qengineering.eu/images/test_parkV5.jpg )
## YoloV8 with the TensorRT framework. <br/>
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)<br/><br/>
A lightweight C++ implementation of YoloV8 running on NVIDIAs TensorRT engine.<br>
No additional libraries are required, just a few lines of code using software, found on every JetPack.<br><br>
For now: https://github.com/akashAD98/yolov8_in_depth<br>
Paper: on Ultralytics TODO list https://github.com/ultralytics/ultralytics<br/><br/>
Specially made for a Jetson Nano see [Q-engineering deep learning examples](https://qengineering.eu/deep-learning-examples-on-raspberry-32-64-os.html)

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
$ mkdir *MyDir* <br/>
$ cd *MyDir* <br/>
$ git clone https://github.com/Qengineering/oloV8-TensorRT-Jetson_Nano.git <br/>
```

------------

## Getting your model.
TensorRT works with `*.engine` models.<br>
The models must be generated by the same version as the TensorRT version on your Jetson, otherwise you run into errors.<br>
That's why we provide the underlying `onnx` models instead of the `engine` models.<br> 
{imb}<br><br>
You need the `trtexec` app to convert the model from onnx to the engine format.<br>
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
Instead of `--fp16`, you could use `--int8`. All 16-bit floating points are now pruned to 8-bit integers, giving you a smaller but less accurate model.<br>
You can run the app once you have your `yolov8s.engine` model.

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
![output image]( https://qengineering.eu/github/test_busV8.webp )

------------

[![paypal](https://qengineering.eu/images/TipJarSmall4.png)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=CPZTM5BB3FCYL) 
