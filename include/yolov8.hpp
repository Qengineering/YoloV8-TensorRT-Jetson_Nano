//
// Created by  triple-Mu     on 24-1-2023.
// Modified by Q-engineering on  6-3-2024,
// Modified by niklasrah     on 9-12-2025,
//
#ifndef DETECT_NORMAL_YOLOV8_HPP
#define DETECT_NORMAL_YOLOV8_HPP

#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"

using namespace det;

class YOLOv8 {
    public:
        explicit YOLOv8(const std::string& engine_file_path);
        ~YOLOv8();

        void MakePipe(bool warmup = true);
        void CopyFromMat(const cv::Mat& image);
        void CopyFromMat(const cv::Mat& image, cv::Size& size);
        void Letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
        void Infer();
        void PostProcess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int num_labels  = 80);
        void DrawObjects(cv::Mat& bgr, const std::vector<Object>& objs);

    private:
        // TensorRT objects
        nvinfer1::ICudaEngine* engine  = nullptr;
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;

        // TensorRT objects
        cudaStream_t stream  = nullptr;

        // Bindings (now name-based)
        std::vector<Binding> input_bindings;
        std::vector<Binding> output_bindings;

        // Buffers
        std::vector<void*> device_ptrs;  // inputs first, then outputs
        std::vector<void*> host_ptrs;    // host pinned outputs only

        // logger
        Logger gLogger{nvinfer1::ILogger::Severity::kERROR};

        // changed to num_iotensors for name-based bindings
        int num_iotensors = 0;
        int num_inputs = 0;
        int num_outputs = 0;

        PreParam pparam;

        // helpers
        size_t get_size_by_dims_local(const nvinfer1::Dims& d) { return static_cast<size_t>(get_size_by_dims(d)); }

};
#endif  // DETECT_NORMAL_YOLOV8_HPP
