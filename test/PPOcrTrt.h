#pragma once
// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>
#include "NvInferPlugin.h"
// cuda include
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <device_launch_parameters.h>
#include <memory>
#include <QtCore/QDebug>
#include <QtCore/QHash>
#include "OCR/include/postprocess_op.h"
#include "OCR/include/utility.h"

using namespace PaddleOCR;

namespace POT {
	class TRTLogger : public nvinfer1::ILogger {
	public:
		inline const char* severity_string(nvinfer1::ILogger::Severity t) {
			switch (t) {
			case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
			case nvinfer1::ILogger::Severity::kERROR:   return "error";
			case nvinfer1::ILogger::Severity::kWARNING: return "warning";
			case nvinfer1::ILogger::Severity::kINFO:    return "info";
			case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
			default: return "unknow";
			}
		};
		virtual void log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
			if (severity <= Severity::kINFO) {
				// 打印带颜色的字符，格式如下：
				// printf("\033[47;33m打印的文本\033[0m");
				// 其中 \033[ 是起始标记
				//      47    是背景颜色
				//      ;     分隔符
				//      33    文字颜色
				//      m     开始标记结束
				//      \033[0m 是终止标记
				// 其中背景颜色或者文字颜色可不写
				// 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
				if (severity == Severity::kWARNING) {
					qDebug() << severity_string(severity) << msg;
					//printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
				}
				else if (severity <= Severity::kERROR) {
					qDebug() << severity_string(severity) << msg;
					//printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
				}
				else {
					qDebug() << severity_string(severity) << msg;
					//printf("%s: %s\n", severity_string(severity), msg);
				}
			}
		};
	};

	struct BuildConfig
	{
		QString engine_file;
		nvinfer1::Dims KMIN;
		nvinfer1::Dims kOPT;
		nvinfer1::Dims kMAX;
		float workSpaceSize;
		nvinfer1::BuilderFlag flag;
	};

	class PPOcrTrtDet
	{
	public:
		explicit PPOcrTrtDet();
		~PPOcrTrtDet();
		bool loadModel(QString fileName);
		std::vector<PaddleOCR::OCRPredictResult> Run(cv::Mat& img);
	public:
		int build_model(QString fileName);
		void preprocess(const cv::Mat& img, cv::Mat& pred_map, cv::Mat& cbuf_map, float& ratio_h, float& ratio_w);
		void post_processor(const cv::Mat& img, std::vector<PaddleOCR::OCRPredictResult>& boxes,
			cv::Mat& pred_map, cv::Mat& bit_map, float& ratio_h, float& ratio_w);
	private:

	private:
		TRTLogger logger;

		//nvinfer1::IRuntime* runtime = nullptr;
		nvinfer1::ICudaEngine* engine = nullptr;
		nvinfer1::IExecutionContext* execution_context = nullptr;

		PaddleOCR::DBPostProcessor post_processor_;

		int limit_side_len = 960;
		double det_db_thresh_ = 0.3;
		double det_db_box_thresh_ = 0.5;
		double det_db_unclip_ratio_ = 1.5;
		std::string det_db_score_mode_ = "slow";
	};

	class PPOcrTrtRec
	{
	public:
		PPOcrTrtRec(const std::string& label_path) {
			this->label_list_ = Utility::ReadDict(label_path);
			this->label_list_.insert(this->label_list_.begin(),
				"#"); // blank char for ctc
			this->label_list_.push_back(" ");
		}
		~PPOcrTrtRec() {
			//if (execution_context)
			//{
			//	execution_context->destroy();
			//}
			//if (engine)
			//{
			//	engine->destroy();
			//}
		}

	public:
		bool loadModel(QString fileName);
		void Run(std::vector<PaddleOCR::OCRPredictResult>ocr_results, const cv::Mat& img,
			std::vector<std::string>& rec_texts_,
			std::vector<float>& rec_text_scores_);
		void Run(const cv::Mat& img,
			std::vector<std::string>& rec_texts_,
			std::vector<float>& rec_text_scores_);
		void Run(std::vector<cv::Mat>& img_list,
			std::vector<std::string>& rec_texts_,
			std::vector<float>& rec_text_scores_);
        void preprocess(const cv::Mat& img, float* input_d, int imgW, int index);
	private:
		int build_model(QString fileName);
		void detect(std::vector<cv::Mat>& img_list, 
			std::vector<std::string>& rec_texts_,
			std::vector<float>& rec_text_scores_);
		TRTLogger logger;
		nvinfer1::ICudaEngine* engine = nullptr;
		nvinfer1::IExecutionContext* execution_context = nullptr;

		int rec_batch_num_ = 6;
		int rec_img_h_ = 48;
		int rec_img_w_ = 320;
		int limit_side_len = 960;
		std::vector<int> rec_image_shape_ = { 3, rec_img_h_, rec_img_w_ };

		std::vector<std::string> label_list_;
	};
}