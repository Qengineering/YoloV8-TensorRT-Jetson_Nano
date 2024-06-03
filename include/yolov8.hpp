//
// Created by  triple-Mu     on 24-1-2023.
// Modified by Q-engineering on  6-3-2024
//
#ifndef DETECT_NORMAL_YOLOV8_HPP
#define DETECT_NORMAL_YOLOV8_HPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"

using namespace det;

class YOLOv8 {
private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
public:
    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;

    PreParam pparam;

public:
    explicit YOLOv8(const std::string& engine_file_path);
    ~YOLOv8();

    void                 MakePipe(bool warmup = true);
    void                 CopyFromMat(const cv::Mat& image);
    void                 CopyFromMat(const cv::Mat& image, cv::Size& size);
    void                 Letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void                 Infer();
    void                 PostProcess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int num_labels  = 80);
    void                 DrawObjects(cv::Mat& bgr, const std::vector<Object>& objs);
};
#endif  // DETECT_NORMAL_YOLOV8_HPP
