#include "PPOcrTrt.h"
#include "common/common.h"
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

namespace POT {
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
	bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
		if (code != cudaSuccess) {
			const char* err_name = cudaGetErrorName(code);
			const char* err_message = cudaGetErrorString(code);
			QString info = QString(u8"错误 %1:%2 %3. 问题:%4, 信息:%5").arg(file).arg(line).arg(op).arg(err_name).arg(err_message);
			qDebug() << info;
			//printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
			return false;
		}
		return true;
	}
	static bool fileExists(const std::string& filename) {
		return (access(filename.c_str(), 0) != -1);
	}
	static std::vector<unsigned char> load_file(const std::string& file) {
		std::ifstream in(file, std::ios::in | std::ios::binary);
		if (!in.is_open())
			return {};

		in.seekg(0, std::ios::end);
		size_t length = in.tellg();

		std::vector<uint8_t> data;
		if (length > 0) {
			in.seekg(0, std::ios::beg);
			data.resize(length);

			in.read((char*)&data[0], length);
		}
		in.close();
		return data;
	}
	int build(BuildConfig& config, TRTLogger& logger)
	{
		auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
		if (!builder)
		{
			return -1;
		}
		auto configEngine = SampleUniquePtr< nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
		if (!configEngine)
		{
			return -2;
		}
		const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
		if (!network)
		{
			return -3;
		}
		auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
		if (!parser)
		{
			return -4;
		}

		if (!parser->parseFromFile(config.engine_file.toStdString().c_str(), 1)) {
			return -5;
		}
		//samplesCommon::enableDLA(builder.get(), config.get(), -1);
		int n = network->getNbInputs();
		qDebug() << u8"模型输入个数：" << n;

		for (size_t i = 0; i < n; i++)
		{
			auto input_tensor = network->getInput(i);
			auto dim = input_tensor->getDimensions();
			auto name = input_tensor->getName();
			qDebug() << u8"第" << i << u8"个输入," << u8"张量维度:" << dim.nbDims << u8"节点名称:" << name;
		}
		// 每个有动态shape的输入都要设置
		auto profile = builder->createOptimizationProfile();
		auto input_tensor_point_coords = network->getInput(0);

		// 配置输入的最小、最优、最大的范围
		profile->setDimensions(input_tensor_point_coords->getName(), nvinfer1::OptProfileSelector::kMIN, config.KMIN);
		profile->setDimensions(input_tensor_point_coords->getName(), nvinfer1::OptProfileSelector::kOPT, config.kOPT);
		profile->setDimensions(input_tensor_point_coords->getName(), nvinfer1::OptProfileSelector::kMAX, config.kMAX);

		if (!profile->isValid())
			return -10;
		configEngine->addOptimizationProfile(profile);

		qDebug() << "Workspace Size:" << config.workSpaceSize / 1024.0f / 1024.0f;
		configEngine->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, config.workSpaceSize);
		configEngine->setPreviewFeature(PreviewFeature::kFASTER_DYNAMIC_SHAPES_0805, true);
		configEngine->setFlag(config.flag);
		

		// CUDA stream used for profiling by the builder.
		auto profileStream = samplesCommon::makeCudaStream();
		if (!profileStream)
		{
			return -6;
		}
		configEngine->setProfileStream(*profileStream);

		SampleUniquePtr<nvinfer1::IHostMemory> plan{ builder->buildSerializedNetwork(*network, *configEngine) };
		if (!plan)
		{
			return -7;
		}

		SampleUniquePtr<IRuntime> runtime{ createInferRuntime(logger) };
		if (!runtime)
		{
			return -8;
		}
		std::shared_ptr<nvinfer1::ICudaEngine> engine = //!< The TensorRT engine used to run the network
			std::shared_ptr<nvinfer1::ICudaEngine>(
				runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
		if (!engine)
		{
			return -9;
		}
		// 将模型序列化，并储存为文件
		nvinfer1::IHostMemory* model_data = engine->serialize();
		auto baseName = config.engine_file.replace(".onnx", ".trt");
		FILE* f = fopen(baseName.toStdString().c_str(), "wb");
		fwrite(model_data->data(), 1, model_data->size(), f);
		fclose(f);
		model_data->destroy();
		return 1;
	}
	POT::PPOcrTrtDet::PPOcrTrtDet()
	{
	}

	POT::PPOcrTrtDet::~PPOcrTrtDet()
	{
		//if (execution_context)
		//{
		//	execution_context->destroy();
		//}
  //      if (engine)
		//{
		//	engine->destroy();
		//}
	}

	int POT::PPOcrTrtDet::build_model(QString fileName)
	{
		BuildConfig config;
		config.engine_file = fileName;
        config.workSpaceSize = 1 << 30;
        config.flag = nvinfer1::BuilderFlag::kFP16;
        config.KMIN = nvinfer1::Dims4{ 1, 3, 32, 64 };
        config.kOPT = nvinfer1::Dims4{ 1, 3, 64, 320 };
        config.kMAX = nvinfer1::Dims4{ 4, 3, limit_side_len, limit_side_len };
		return build(config, logger);
	}
	
	bool POT::PPOcrTrtDet::loadModel(QString fileName)
	{
		if (!fileExists(fileName.toStdString().c_str()))
		{
			auto onnx_file = fileName;
			if (build_model(onnx_file.replace(".trt", ".onnx")) < 0) {
				return false;
			}
		}
		else
		{
			auto engine_data = load_file(fileName.toStdString().c_str());
			// 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger：
			SampleUniquePtr<IRuntime> runtime{ createInferRuntime(logger) };
			//runtime = nvinfer1::createInferRuntime(logger);

			engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
		}

		if (engine == nullptr) {
			return false;
		}
		execution_context = engine->createExecutionContext();

		return true;
	}

	__global__ void preprocessImageKernel(
		const float* input,
		float* output,
		int inputWidth,
		int inputHeight,
		int outputWidth,
		int outputHeight,
		float* input_ptr)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < outputWidth && y < outputHeight) {
			// Bilinear interpolation for resizing
			float u = (static_cast<float>(x) + 0.5f) * inputWidth / outputWidth - 0.5f;
			float v = (static_cast<float>(y) + 0.5f) * inputHeight / outputHeight - 0.5f;

			int u0 = static_cast<int>(u);
			int v0 = static_cast<int>(v);
			int u1 = u0 + 1;
			int v1 = v0 + 1;

			float a = u - u0;
			float b = v - v0;

			float r00 = (u0 >= 0 && u0 < inputWidth&& v0 >= 0 && v0 < inputHeight) ? input[3 * (v0 * inputWidth + u0) + 2] : 0.0f;
			float g00 = (u0 >= 0 && u0 < inputWidth&& v0 >= 0 && v0 < inputHeight) ? input[3 * (v0 * inputWidth + u0) + 1] : 0.0f;
			float b00 = (u0 >= 0 && u0 < inputWidth&& v0 >= 0 && v0 < inputHeight) ? input[3 * (v0 * inputWidth + u0) + 0] : 0.0f;

			float r01 = (u0 >= 0 && u0 < inputWidth&& v1 >= 0 && v1 < inputHeight) ? input[3 * (v1 * inputWidth + u0) + 2] : 0.0f;
			float g01 = (u0 >= 0 && u0 < inputWidth&& v1 >= 0 && v1 < inputHeight) ? input[3 * (v1 * inputWidth + u0) + 1] : 0.0f;
			float b01 = (u0 >= 0 && u0 < inputWidth&& v1 >= 0 && v1 < inputHeight) ? input[3 * (v1 * inputWidth + u0) + 0] : 0.0f;

			float r10 = (u1 >= 0 && u1 < inputWidth&& v0 >= 0 && v0 < inputHeight) ? input[3 * (v0 * inputWidth + u1) + 2] : 0.0f;
			float g10 = (u1 >= 0 && u1 < inputWidth&& v0 >= 0 && v0 < inputHeight) ? input[3 * (v0 * inputWidth + u1) + 1] : 0.0f;
			float b10 = (u1 >= 0 && u1 < inputWidth&& v0 >= 0 && v0 < inputHeight) ? input[3 * (v0 * inputWidth + u1) + 0] : 0.0f;

			float r11 = (u1 >= 0 && u1 < inputWidth&& v1 >= 0 && v1 < inputHeight) ? input[3 * (v1 * inputWidth + u1) + 2] : 0.0f;
			float g11 = (u1 >= 0 && u1 < inputWidth&& v1 >= 0 && v1 < inputHeight) ? input[3 * (v1 * inputWidth + u1) + 1] : 0.0f;
			float b11 = (u1 >= 0 && u1 < inputWidth&& v1 >= 0 && v1 < inputHeight) ? input[3 * (v1 * inputWidth + u1) + 0] : 0.0f;

			float r1 = (1 - a) * ((1 - b) * r00 + b * r01) + a * ((1 - b) * r10 + b * r11);
			float g1 = (1 - a) * ((1 - b) * g00 + b * g01) + a * ((1 - b) * g10 + b * g11);
			float b1 = (1 - a) * ((1 - b) * b00 + b * b01) + a * ((1 - b) * b10 + b * b11);

			// 原图
			//output[3 * (y * outputWidth + x) + 0] = b1;
			//output[3 * (y * outputWidth + x) + 1] = g1;
			//output[3 * (y * outputWidth + x) + 2] = r1;

			// Normalize and preprocess
			r1 = (r1 / 255.0f - 0.485) / 0.229;
			g1 = (g1 / 255.0f - 0.456) / 0.224;
			b1 = (b1 / 255.0f - 0.406) / 0.225;

			// 模型输入
			int imgArea = outputWidth * outputHeight;

			int bias = y * outputWidth + x;

			float* phost_b = input_ptr + imgArea * 0;
			float* phost_g = input_ptr + imgArea * 1;
			float* phost_r = input_ptr + imgArea * 2;

			*(phost_b + bias) = r1;
			*(phost_g + bias) = g1;
			*(phost_r + bias) = b1;
		}
	}

	__global__ void maskPostprocessing(float* input, int thresh,int width, int height)
	{
		int dx = blockDim.x * blockIdx.x + threadIdx.x;
		int dy = blockDim.y * blockIdx.y + threadIdx.y;
		if (dx >= width || dy >= height)  return;
		// 计算像素在图像中的索引
		int index = dy * width + dx;
		//input[index] *= 255;
		//input[index] = (int)input[index] > thresh * 255 ? 255 : 0;
		input[index] = input[index] > thresh ? 255 : 0;
	}

	void preprocessImageCUDA(const cv::Mat& input, cv::Mat& output, float* d_input_ptr) {
		int inputWidth = input.cols;
		int inputHeight = input.rows;
		int outputWidth = output.cols;
		int outputHeight = output.rows;

		float* d_input;
		float* d_output;

		cudaMalloc(&d_input, 3 * inputWidth * inputHeight * sizeof(float));
		cudaMalloc(&d_output, 3 * outputWidth * outputHeight * sizeof(float));
		

		cv::Mat inputFloat;
		input.convertTo(inputFloat, CV_32FC3, 1.0f);

		cudaMemcpy(d_input, inputFloat.data, 3 * inputWidth * inputHeight * sizeof(float), cudaMemcpyHostToDevice);

		dim3 blockSize(16, 16);
		dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x, (outputHeight + blockSize.y - 1) / blockSize.y);

		preprocessImageKernel << <gridSize, blockSize >> > (d_input, d_output, inputWidth, inputHeight, outputWidth, outputHeight, d_input_ptr);
		cudaMemcpy(output.data, d_output, 3 * outputWidth * outputHeight * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(d_input);
		cudaFree(d_output);
	}

	void POT::PPOcrTrtDet::preprocess(const cv::Mat& img, cv::Mat& pred_map, cv::Mat& bit_map, float& ratio_h,float& ratio_w)
	{
		
		int w = img.cols;
		int h = img.rows;
		float ratio = 1.f;

		int max_wh = std::max(h, w);
		if (max_wh > limit_side_len) {
			if (h > w) {
				ratio = float(limit_side_len) / float(h);
			}
			else {
				ratio = float(limit_side_len) / float(w);
			}
		}


		int resize_h = int(float(h) * ratio);
		int resize_w = int(float(w) * ratio);

		resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
		resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);

		cv::Mat resize_img(cv::Size(resize_w, resize_h), CV_32FC3);
		float* int_d;
		checkRuntime(cudaMalloc((void**)&int_d, resize_h * resize_w * 3 * sizeof(float))); // 在GPU上开辟空间
        preprocessImageCUDA(img, resize_img, int_d);
		
		execution_context->setBindingDimensions(0, nvinfer1::Dims4(1, 3, resize_h, resize_w));
		float* out_d;
		checkRuntime(cudaMalloc(&out_d, resize_h * resize_w * 3 * sizeof(float))); // 在GPU上开辟空间
		std::vector<void*> enginBuffs = { int_d, out_d };
		execution_context->executeV2(enginBuffs.data());

		pred_map = cv::Mat(resize_h, resize_w, CV_32F);
		checkRuntime(cudaMemcpy(pred_map.data, out_d, sizeof(float) * resize_h * resize_w, cudaMemcpyDeviceToHost));

		//auto dim_out = execution_context->getBindingDimensions(1);
		//qDebug()<<dim_out.d[0]<< dim_out.d[1]<<dim_out.d[2]<<dim_out.d[3];
		dim3 blockSize(16, 16);
		dim3 gridSize((resize_w + blockSize.x - 1) / blockSize.x, (resize_h + blockSize.y - 1) / blockSize.y);
		maskPostprocessing << <gridSize, blockSize, 0, nullptr >> > (out_d, det_db_thresh_, resize_w, resize_h);

		bit_map = cv::Mat(resize_h, resize_w, CV_32F);
		checkRuntime(cudaMemcpy(bit_map.data, out_d, sizeof(float) * resize_h * resize_w, cudaMemcpyDeviceToHost));
		bit_map.convertTo(bit_map, CV_8UC1);
		ratio_h = float(resize_h) / float(h);
		ratio_w = float(resize_w) / float(w);

		checkRuntime(cudaFree(out_d));
		checkRuntime(cudaFree(int_d));

	}
	
	
	void POT::PPOcrTrtDet::post_processor(const cv::Mat& img, std::vector<PaddleOCR::OCRPredictResult>& ocr_results, cv::Mat& pred_map,
		cv::Mat& bit_map, float& ratio_h, float& ratio_w)
	{
		auto boxes = post_processor_.BoxesFromBitmap(
			pred_map, bit_map, this->det_db_box_thresh_, this->det_db_unclip_ratio_,
			this->det_db_score_mode_);
		cv::Mat srcimg;
		img.copyTo(srcimg);
		boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
		for (int i = 0; i < boxes.size(); i++) {
			PaddleOCR::OCRPredictResult res;
			res.box = boxes[i];
			ocr_results.push_back(res);
		}
		// sort boex from top to bottom, from left to right
		PaddleOCR::Utility::sorted_boxes(ocr_results);
	}

	std::vector<PaddleOCR::OCRPredictResult> POT::PPOcrTrtDet::Run(cv::Mat& src)
	{
		float ratio_h, ratio_w;
		cv::Mat pred_map, bit_map;
		preprocess(src, pred_map, bit_map, ratio_h, ratio_w);
		std::vector<PaddleOCR::OCRPredictResult> ocr_results;
		post_processor(src, ocr_results, pred_map, bit_map, ratio_h, ratio_w);
        return ocr_results;
	}

	int POT::PPOcrTrtRec::build_model(QString fileName)
	{
		BuildConfig config;
		config.engine_file = fileName;
		config.workSpaceSize = 1 << 30;
		config.flag = nvinfer1::BuilderFlag::kFP16;
		config.KMIN = nvinfer1::Dims4{ 1, 3, 48, 64 };
		config.kOPT = nvinfer1::Dims4{ 6, 3, 48, 320 };
		config.kMAX = nvinfer1::Dims4{ 10, 3, 48, limit_side_len };
		return build(config, logger);
	}

	bool POT::PPOcrTrtRec::loadModel(QString fileName)
	{
		if (!fileExists(fileName.toStdString().c_str()))
		{
			auto onnx_file = fileName;
			if (build_model(onnx_file.replace(".trt", ".onnx")) < 0) {
				return false;
			}
		}
		auto engine_data = load_file(fileName.toStdString().c_str());
		// 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger：
		SampleUniquePtr<IRuntime> runtime{ createInferRuntime(logger) };
		//runtime = nvinfer1::createInferRuntime(logger);

		engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
		if (engine == nullptr) {
            qDebug() << "load model failed";
			return false;
		}
		execution_context = engine->createExecutionContext();
		return true;
	}

	__global__ void recPreprocessKernel(unsigned char* input, float* output, int inputWidth, int inputHeight, int resize_w, int resize_h, int right)
	{
        int dx = blockDim.x * blockIdx.x + threadIdx.x;
        int dy = blockDim.y * blockIdx.y + threadIdx.y;
        if (dx >= (resize_w+right) || dy >= resize_h)  return;

		float b1= 0.f, g1= 0.f, r1= 0.f;
		if (dx < resize_w && dy < resize_h)
		{
			// Bilinear interpolation for resizing
			float u = (static_cast<float>(dx) + 0.5f) * inputWidth / resize_w - 0.5f;
			float v = (static_cast<float>(dy) + 0.5f) * inputHeight / resize_h - 0.5f;

			int u0 = static_cast<int>(u);
			int v0 = static_cast<int>(v);
			int u1 = u0 + 1;
			int v1 = v0 + 1;

			float a = u - u0;
			float b = v - v0;

			float r00 = (u0 >= 0 && u0 < inputWidth&& v0 >= 0 && v0 < inputHeight) ? input[3 * (v0 * inputWidth + u0) + 2] : 0.0f;
			float g00 = (u0 >= 0 && u0 < inputWidth&& v0 >= 0 && v0 < inputHeight) ? input[3 * (v0 * inputWidth + u0) + 1] : 0.0f;
			float b00 = (u0 >= 0 && u0 < inputWidth&& v0 >= 0 && v0 < inputHeight) ? input[3 * (v0 * inputWidth + u0) + 0] : 0.0f;

			float r01 = (u0 >= 0 && u0 < inputWidth&& v1 >= 0 && v1 < inputHeight) ? input[3 * (v1 * inputWidth + u0) + 2] : 0.0f;
			float g01 = (u0 >= 0 && u0 < inputWidth&& v1 >= 0 && v1 < inputHeight) ? input[3 * (v1 * inputWidth + u0) + 1] : 0.0f;
			float b01 = (u0 >= 0 && u0 < inputWidth&& v1 >= 0 && v1 < inputHeight) ? input[3 * (v1 * inputWidth + u0) + 0] : 0.0f;

			float r10 = (u1 >= 0 && u1 < inputWidth&& v0 >= 0 && v0 < inputHeight) ? input[3 * (v0 * inputWidth + u1) + 2] : 0.0f;
			float g10 = (u1 >= 0 && u1 < inputWidth&& v0 >= 0 && v0 < inputHeight) ? input[3 * (v0 * inputWidth + u1) + 1] : 0.0f;
			float b10 = (u1 >= 0 && u1 < inputWidth&& v0 >= 0 && v0 < inputHeight) ? input[3 * (v0 * inputWidth + u1) + 0] : 0.0f;

			float r11 = (u1 >= 0 && u1 < inputWidth&& v1 >= 0 && v1 < inputHeight) ? input[3 * (v1 * inputWidth + u1) + 2] : 0.0f;
			float g11 = (u1 >= 0 && u1 < inputWidth&& v1 >= 0 && v1 < inputHeight) ? input[3 * (v1 * inputWidth + u1) + 1] : 0.0f;
			float b11 = (u1 >= 0 && u1 < inputWidth&& v1 >= 0 && v1 < inputHeight) ? input[3 * (v1 * inputWidth + u1) + 0] : 0.0f;

			r1 = (1 - a) * ((1 - b) * r00 + b * r01) + a * ((1 - b) * r10 + b * r11);
			g1 = (1 - a) * ((1 - b) * g00 + b * g01) + a * ((1 - b) * g10 + b * g11);
			b1 = (1 - a) * ((1 - b) * b00 + b * b01) + a * ((1 - b) * b10 + b * b11);
		}

		int index = dy * (resize_w+ right) + dx;
		// 原图
		//output[3 * index + 0] = (r1 / 255 - 0.5) / 0.5;
		//output[3 * index + 1] = (g1 / 255 - 0.5) / 0.5;
		//output[3 * index + 2] = (b1 / 255 - 0.5) / 0.5;
		// 模型输入
		int outputWidth = resize_w + right;
		int imgArea = outputWidth * resize_h;

		int bias = dy * outputWidth + dx;

		float* phost_b = output + imgArea * 0;
		float* phost_g = output + imgArea * 1;
		float* phost_r = output + imgArea * 2;

		*(phost_b + bias) = (r1 / 255 - 0.5) / 0.5;
		*(phost_g + bias) = (g1 / 255 - 0.5) / 0.5;
		*(phost_r + bias) = (b1 / 255 - 0.5) / 0.5;
	}

	void POT::PPOcrTrtRec::preprocess(const cv::Mat& img, float* input_d, int imgW, int index)
	{
		int imgH = rec_img_h_;

		float ratio = float(img.cols) / float(img.rows);
		int resize_w;

		if (ceilf(imgH * ratio) > imgW)
			resize_w = imgW;
		else
			resize_w = int(ceilf(imgH * ratio));

		unsigned char* int_d;
        checkRuntime(cudaMalloc((void**)&int_d, img.cols * img.rows * 3 * sizeof(unsigned char)));
		cudaMemcpy(int_d, img.data, 3 * img.cols * img.rows * sizeof(unsigned char), cudaMemcpyHostToDevice);

		float* out_d;
        checkRuntime(cudaMalloc(&out_d, imgH * imgW * 3 * sizeof(float)));
        dim3 blockSize(16, 16);
        dim3 gridSize((imgW + blockSize.x - 1) / blockSize.x, (imgH + blockSize.y - 1) / blockSize.y);
        recPreprocessKernel << <gridSize, blockSize, 0, nullptr >> > (int_d, out_d, img.cols, img.rows, resize_w, imgH, int(imgW - resize_w));
		//resize_img = cv::Mat(imgH, imgW, CV_32FC3);
		//checkRuntime(cudaMemcpy(resize_img.data, out_d, sizeof(float) * imgH * imgW*3, cudaMemcpyDeviceToHost));
        checkRuntime(cudaMemcpy(input_d+ index* imgH * imgW * 3, out_d, sizeof(float) * imgH * imgW*3, cudaMemcpyDeviceToDevice));
        checkRuntime(cudaFree(int_d));
        checkRuntime(cudaFree(out_d));
	}

	__global__ void findRowMaxAndIndex(float* input, float* maxValues, int* maxIndices, int batch, int rows, int cols) {
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		int batchIndex = blockIdx.y;

		if (row < rows) {
			float maxVal = -FLT_MAX;
			int maxIdx = 0;
			int baseIndex = batchIndex * rows * cols + row * cols;

			for (int col = 0; col < cols; ++col) {
				float val = input[baseIndex + col];
				if (val > maxVal) {
					maxVal = val;
					maxIdx = col;
				}
			}

			maxValues[batchIndex * rows + row] = maxVal;
			maxIndices[batchIndex * rows + row] = maxIdx;
		}
	}

	void POT::PPOcrTrtRec::detect(std::vector<cv::Mat>& img_list, std::vector<std::string>& rec_texts_, std::vector<float>& rec_text_scores_)
	{
		std::vector<std::string> rec_texts(img_list.size(), "");
		std::vector<float> rec_text_scores(img_list.size(), 0);
		int img_num = img_list.size();
		std::vector<float> width_list;
		for (int i = 0; i < img_num; i++) {
			width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
		}
		std::vector<int> indices = Utility::argsort(width_list);
		for (int beg_img_no = 0; beg_img_no < img_num; beg_img_no += this->rec_batch_num_) {
			int end_img_no = std::min(img_num, beg_img_no + this->rec_batch_num_);
			int batch_num = end_img_no - beg_img_no;
			int imgH = this->rec_image_shape_[1];
			int imgW = this->rec_image_shape_[2];
			float max_wh_ratio = imgW * 1.0 / imgH;
			for (int ino = beg_img_no; ino < end_img_no; ino++) {
				int h = img_list[indices[ino]].rows;
				int w = img_list[indices[ino]].cols;
				float wh_ratio = w * 1.0 / h;
				max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
			}

			int batch_width = std::min(int(imgH * max_wh_ratio), limit_side_len);
			float* input_d;
			checkRuntime(cudaMalloc(&input_d, batch_num * imgH * batch_width * 3 * sizeof(float)));
			int index = 0;
			for (int ino = beg_img_no; ino < end_img_no; ino++) {
				cv::Mat srcimg;
				img_list[indices[ino]].copyTo(srcimg);
				preprocess(srcimg, input_d, batch_width, index++);
			}

			execution_context->setBindingDimensions(0, nvinfer1::Dims4(batch_num, 3, imgH, batch_width));
			auto dim_out = execution_context->getBindingDimensions(1);
			float* output_d;
			checkRuntime(cudaMalloc(&output_d, batch_num * dim_out.d[1] * dim_out.d[2] * sizeof(float)));
			std::vector<void*> enginBuffs = { input_d, output_d };
			execution_context->executeV2(enginBuffs.data());


			float* d_maxValues;
			int* d_maxIndices;

			cudaMalloc(&d_maxValues, sizeof(float) * batch_num * dim_out.d[1]);
			cudaMalloc(&d_maxIndices, sizeof(int) * batch_num * dim_out.d[1]);
			dim3 blockSize(dim_out.d[1]);
			dim3 gridSize(1, batch_num);

			findRowMaxAndIndex << <gridSize, blockSize >> > (output_d, d_maxValues, d_maxIndices, batch_num, dim_out.d[1], dim_out.d[2]);

			std::vector<float> h_maxValues(batch_num * dim_out.d[1], 0.0f);
			std::vector<int> h_maxIndices(batch_num * dim_out.d[1], 0);

			cudaMemcpy(h_maxValues.data(), d_maxValues, sizeof(float) * batch_num * dim_out.d[1], cudaMemcpyDeviceToHost);
			cudaMemcpy(h_maxIndices.data(), d_maxIndices, sizeof(int) * batch_num * dim_out.d[1], cudaMemcpyDeviceToHost);

			cudaFree(d_maxValues);
			cudaFree(d_maxIndices);
			checkRuntime(cudaFree(input_d));
			checkRuntime(cudaFree(output_d));
			for (int m = 0; m < dim_out.d[0]; m++) {
				std::string str_res;
				int argmax_idx;
				int last_index = 0;
				float score = 0.f;
				int count = 0;
				float max_value = 0.0f;

				for (int n = 0; n < dim_out.d[1]; n++) {
					argmax_idx = h_maxIndices[m * dim_out.d[1] + n];
					max_value = h_maxValues[m * dim_out.d[1] + n];

					if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
						score += max_value;
						count += 1;
						str_res += label_list_[argmax_idx];
					}
					last_index = argmax_idx;
				}
				score /= count;
				if (std::isnan(score)) {
					continue;
				}
				if (score>0.4)
				{
					rec_texts[indices[beg_img_no + m]] = str_res;
					rec_text_scores[indices[beg_img_no + m]] = score;
				}
			}
		}
		rec_texts_ = rec_texts;
		rec_text_scores_ = rec_text_scores;
	}

	void POT::PPOcrTrtRec::Run(std::vector<PaddleOCR::OCRPredictResult> ocr_result, const cv::Mat& img,
		std::vector<std::string>& rec_texts_,
		std::vector<float>& rec_text_scores_)
	{
		// crop image
		std::vector<cv::Mat> img_list;
		for (int j = 0; j < ocr_result.size(); j++) {
			cv::Mat crop_img;
			crop_img = Utility::GetRotateCropImage(img, ocr_result[j].box);
			img_list.push_back(crop_img);
		}
		detect(img_list, rec_texts_, rec_text_scores_);
	}
	void POT::PPOcrTrtRec::Run(const cv::Mat& img, std::vector<std::string>& rec_texts_, std::vector<float>& rec_text_scores_)
	{
		std::vector<cv::Mat> img_list = { img };
		detect(img_list, rec_texts_, rec_text_scores_);
	}
	void POT::PPOcrTrtRec::Run(std::vector<cv::Mat>& img_list, std::vector<std::string>& rec_texts_, std::vector<float>& rec_text_scores_)
	{
		detect(img_list, rec_texts_, rec_text_scores_);
	}
}