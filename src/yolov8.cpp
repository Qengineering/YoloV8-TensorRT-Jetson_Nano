//
// Created by  triple-Mu     on 24-1-2023.
// Modified by Q-engineering on  6-3-2024,
// Modified by niklasrah     on 9-12-2025,
//

#include "yolov8.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>

//----------------------------------------------------------------------------------------
//using namespace det;
//----------------------------------------------------------------------------------------
const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};
//----------------------------------------------------------------------------------------
YOLOv8::YOLOv8(const std::string& engine_file_path)
{
    // Read engine file
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> trtModelStream(static_cast<size_t>(size));
    file.read(trtModelStream.data(), size);
    file.close();

    // init plugins & runtime
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    // TRT10: deserializeCudaEngine has a third parameter (IHostMemoryProvider* or nullptr)
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream.data(), static_cast<size_t>(size));
    assert(this->engine != nullptr);

    // create context
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    // create stream
    CHECK(cudaStreamCreate(&this->stream));

    // discover I/O tensors (name-based)
    this->num_iotensors = this->engine->getNbIOTensors();
    for (int i = 0; i < this->num_iotensors; ++i) {
        const char* tname = this->engine->getIOTensorName(i);
        assert(tname != nullptr);
        nvinfer1::TensorIOMode mode = this->engine->getTensorIOMode(tname);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(tname);
        nvinfer1::Dims dims = this->engine->getTensorShape(tname);  // static shape in engine or profile shape

        det::Binding b;
        b.name = std::string(tname);
        b.dsize = static_cast<size_t>(type_to_size(dtype));
        b.dims = dims;
        b.size = get_size_by_dims_local(dims);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_bindings.push_back(b);
            ++this->num_inputs;
        } else {
            output_bindings.push_back(b);
            ++this->num_outputs;
        }
    }

}
//----------------------------------------------------------------------------------------
YOLOv8::~YOLOv8()
{
    delete this->context;
    delete this->engine;
    delete this->runtime;
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}
//----------------------------------------------------------------------------------------
void YOLOv8::MakePipe(bool warmup)
{
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

    for (auto& inb : this->input_bindings) {
        void* d_ptr = nullptr;
        size_t bytes = inb.size * inb.dsize;
#if (CUDART_VERSION < 11000)
        CHECK(cudaMalloc(&d_ptr, bytes));
#else
        CHECK(cudaMallocAsync(&d_ptr, bytes, this->stream));
#endif
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& outb : this->output_bindings) {
        void* d_ptr = nullptr;
        void* h_ptr = nullptr;
        size_t bytes = outb.size * outb.dsize;
#if (CUDART_VERSION < 11000)
        CHECK(cudaMalloc(&d_ptr, bytes));
#else
        CHECK(cudaMallocAsync(&d_ptr, bytes, this->stream));
#endif
        CHECK(cudaHostAlloc(&h_ptr, bytes, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    // Map buffers to tensor names (required for TRT10)
    int devIndex = 0;

    // Assign input tensor addresses
    for (size_t i = 0; i < input_bindings.size(); ++i) {
        const char* name = input_bindings[i].name.c_str();
        void* d_ptr = device_ptrs[devIndex++];
        context->setTensorAddress(name, d_ptr);
    }

    // Assign output tensor addresses
    for (size_t i = 0; i < output_bindings.size(); ++i) {
        const char* name = output_bindings[i].name.c_str();
        void* d_ptr = device_ptrs[devIndex++];
        context->setTensorAddress(name, d_ptr);
    }

    // optional warmup
    if (warmup) {
        for (int k = 0; k < 5; ++k) {
            // zero input buffers and copy
            for (size_t i = 0; i < this->input_bindings.size(); ++i) {
                size_t bytes = this->input_bindings[i].size * this->input_bindings[i].dsize;
                std::vector<char> zeros(bytes, 0);
                CHECK(cudaMemcpyAsync(this->device_ptrs[i], zeros.data(), bytes, cudaMemcpyHostToDevice, this->stream));
            }
            this->Infer();
        }
    }
}
//----------------------------------------------------------------------------------------
void YOLOv8::Letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = static_cast<float>(image.rows);
    float       width  = static_cast<float>(image.cols);

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
}
//----------------------------------------------------------------------------------------
void YOLOv8::CopyFromMat(const cv::Mat& image)
{
    cv::Mat nchw;
    // assume single input
    assert(this->input_bindings.size() > 0);
    auto& in_binding = this->input_bindings[0];
    // Get target width/height from input binding dims (expect dims format: {N,C,H,W} or nbDims==4)
    int width = (in_binding.dims.nbDims >= 4) ? in_binding.dims.d[3] : 640;
    int height = (in_binding.dims.nbDims >= 4) ? in_binding.dims.d[2] : 640;
    cv::Size size{width, height};
    this->Letterbox(image, nchw, size);

    // set input shape by name (TensorRT 10 name-based API)
    nvinfer1::Dims4 shape{1, 3, height, width};
    this->context->setInputShape(in_binding.name.c_str(), shape);

    size_t bytes = nchw.total() * nchw.elemSize();
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(), bytes, cudaMemcpyHostToDevice, this->stream));
}
//----------------------------------------------------------------------------------------
void YOLOv8::CopyFromMat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->Letterbox(image, nchw, size);

    assert(this->input_bindings.size() > 0);
    auto& in_binding = this->input_bindings[0];
    nvinfer1::Dims4 shape{1, 3, size.height, size.width};
    this->context->setInputShape(in_binding.name.c_str(), shape);

    size_t bytes = nchw.total() * nchw.elemSize();
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(), bytes, cudaMemcpyHostToDevice, this->stream));
}
//----------------------------------------------------------------------------------------
void YOLOv8::Infer()
{
    // On TRT10 use enqueueV3 (name-based tensors still require raw device pointer array)
    this->context->enqueueV3(this->stream);
    // copy outputs back to host; outputs start at index num_inputs
    for (int i = 0; i < this->num_outputs; ++i) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        void* devPtr = this->device_ptrs[i + this->num_inputs];
        void* hostPtr = this->host_ptrs[i];
        CHECK(cudaMemcpyAsync(hostPtr, devPtr, osize, cudaMemcpyDeviceToHost, this->stream));
    }
    CHECK(cudaStreamSynchronize(this->stream));
}
//----------------------------------------------------------------------------------------
void YOLOv8::PostProcess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int num_labels)
{
    objs.clear();
    assert(this->output_bindings.size() > 0);
    auto num_channels = this->output_bindings[0].dims.nbDims > 1 ? this->output_bindings[0].dims.d[1] : 85;  // fallback
    auto num_anchors = this->output_bindings[0].dims.nbDims > 2 ? this->output_bindings[0].dims.d[2] : (this->output_bindings[0].size / num_channels);

    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    std::vector<cv::Rect> bboxes;
    std::vector<float>    scores;
    std::vector<int>      labels;
    std::vector<int>      indices;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto  row_ptr    = output.row(i).ptr<float>();
        auto  bboxes_ptr = row_ptr;
        auto  scores_ptr = row_ptr + 4;
        auto  max_s_ptr  = std::max_element(scores_ptr, scores_ptr + num_labels);
        float score      = *max_s_ptr;
        if (score > score_thres) {
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            int label = max_s_ptr - scores_ptr;
            cv::Rect_<float> bbox;
            bbox.x      = x0;
            bbox.y      = y0;
            bbox.width  = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
    }

#ifdef BATCHED_NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        objs.push_back(obj);
        cnt += 1;
    }
}
//----------------------------------------------------------------------------------------
void YOLOv8::DrawObjects(cv::Mat& bgr, const std::vector<Object>& objs)
{
    char text[256];

    for (auto& obj : objs) {
        cv::rectangle(bgr, obj.rect, cv::Scalar(255, 0, 0));

        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y - label_size.height - baseLine;
        if (y < 0)        y = 0;
        if (y > bgr.rows) y = bgr.rows;
        if (x + label_size.width > bgr.cols) x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);
        cv::putText(bgr, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}
//----------------------------------------------------------------------------------------
