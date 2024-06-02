//
// Created Q-engineering 5-31-2024
//
#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"

using namespace std;
using namespace cv;

cv::Size im_size(640, 640);
const int      num_labels  = 80;
const int      topk        = 100;
const float    score_thres = 0.25f;
const float    iou_thres   = 0.65f;

int main(int argc, char** argv)
{
    float    f;
    float    FPS[16];
    int      i, Fcnt=0;
    cv::Mat  image;
    std::chrono::steady_clock::time_point Tbegin, Tend;

    if (argc < 3) {
        fprintf(stderr,"Usage: ./YoloV8_RT [model_trt.engine] [image or video path] \n");
        return -1;
    }
    const string engine_file_path = argv[1];
    const string imagepath = argv[2];

    for(i=0;i<16;i++) FPS[i]=0.0;

    cout << "Set CUDA...\n" << endl;

    cudaSetDevice(0);

    cout << "Loading TensorRT model " << engine_file_path << endl;
    cout << "\nWait a second...." << std::flush;
    auto yolov8 = new YOLOv8(engine_file_path);

    cout << "\rLoading the pipe... " << string(10, ' ')<< "\n\r" ;
    cout << endl;
    yolov8->MakePipe(true);

    while(1){
        image = cv::imread(imagepath);
        yolov8->CopyFromMat(image, im_size);

        std::vector<Object> objs;

        Tbegin = std::chrono::steady_clock::now();
        yolov8->Infer();
        Tend = std::chrono::steady_clock::now();

        yolov8->PostProcess(objs, score_thres, iou_thres, topk, num_labels);
        yolov8->DrawObjects(image, objs);

        //calculate frame rate
        f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        putText(image, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255));

        //show output
        imshow("Jetson Orin Nano- 8 Mb RAM", image);
        char esc = cv::waitKey(1);
        if(esc == 27) break;
//      imwrite("./out.jpg", image);
    }
    cv::destroyAllWindows();

    delete yolov8;

    return 0;
}
