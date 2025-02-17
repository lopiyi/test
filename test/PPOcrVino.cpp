#include "PPOcrVino.h"
#include <opencv2/opencv.hpp>
#include <QDebug>

using namespace cv;
namespace POV
{
	bool POV::PPOcrVinoDet::loadModel(const QString modelPath)
	{
		try
		{
			ov::Core core;
			//ov::CompiledModel compiled_model = core.compile_model(modelPath.toStdString(), "CPU");
			auto model = core.read_model(modelPath.toStdString());
			//qDebug() << model->input(0).get_partial_shape().is_dynamic();
			ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
			infer_request = compiled_model.create_infer_request();
		}
		catch (const ov::Exception& e)
		{
			qDebug() << u8"错误："<<QString::fromStdString(e.default_msg) << e.what();
            return false;
		}
		return true;
	}

	std::vector<PaddleOCR::OCRPredictResult> POV::PPOcrVinoDet::Run(cv::Mat& src)
	{
		float ratio_h, ratio_w;
		cv::Mat pred_map, bit_map;
		preprocess(src, pred_map, bit_map, ratio_h, ratio_w);
		std::vector<PaddleOCR::OCRPredictResult> ocr_results;
		post_processor(src, ocr_results, pred_map, bit_map, ratio_h, ratio_w);
		return ocr_results;
	}

	void POV::PPOcrVinoDet::preprocess(const cv::Mat& src, cv::Mat& pred_map, cv::Mat& bit_map, float& ratio_h, float& ratio_w)
	{
		int w = src.cols;
		int h = src.rows;

		int resize_h = std::max(int(round(float(h) / 32) * 32), 32);
		int resize_w = std::max(int(round(float(w) / 32) * 32), 32);

		cv::Mat resize_img;
		cv::resize(src, resize_img, cv::Size(resize_w, resize_h));

		ratio_h = float(resize_h) / float(h);
		ratio_w = float(resize_w) / float(w);

		std::vector<float> input(1 * 3 * resize_h * resize_w, 0.0f);
		auto pimage = resize_img.data;
		int image_area = resize_img.rows * resize_img.cols;
		float* input_data_host = input.data();
		float* phost_r = input_data_host + image_area * 0;
		float* phost_g = input_data_host + image_area * 1;
		float* phost_b = input_data_host + image_area * 2;
		for (int i = 0; i < image_area; ++i, pimage += 3) {
			// 注意这里的顺序rgb调换了
			*phost_b++ = (pimage[0] / 255. - 0.406f) / 0.225f;
			*phost_g++ = (pimage[1] / 255. - 0.456f) / 0.224f;
			*phost_r++ = (pimage[2] / 255. - 0.485f) / 0.229f;
		}
		auto input_tensor = ov::Tensor(infer_request.get_input_tensor(0).get_element_type(),
			{ 1, 3, static_cast<unsigned long long>(resize_h), static_cast<unsigned long long>(resize_w) }, input.data());
		infer_request.set_input_tensor(input_tensor);
		infer_request.infer();
		ov::Tensor output_tensor = infer_request.get_output_tensor();
		const float* prob = (float*)output_tensor.data();
		const ov::Shape outputDims = output_tensor.get_shape();
		size_t numRows = outputDims[2];
		size_t numCols = outputDims[3];

		int n2 = outputDims[2];
		int n3 = outputDims[3];
		int n = n2 * n3;

		Mat cbuf_map(n2, n3, CV_32FC1, (unsigned char*)prob);
		cbuf_map *= 255;
		pred_map = Mat(n2, n3, CV_32F, (float*)prob);

		const double threshold = det_db_thresh_ * 255;

		cv::threshold(cbuf_map, bit_map, threshold, 255, cv::THRESH_BINARY);
		bit_map.convertTo(bit_map, CV_8UC1);
	}

	void POV::PPOcrVinoDet::post_processor(const cv::Mat& src, std::vector<PaddleOCR::OCRPredictResult>& ocr_results, cv::Mat& pred_map, cv::Mat& bit_map, float& ratio_h, float& ratio_w)
	{
		auto boxes = post_processor_.BoxesFromBitmap(
			pred_map, bit_map, this->det_db_box_thresh_, this->det_db_unclip_ratio_,
			this->det_db_score_mode_);
		cv::Mat srcimg;
		src.copyTo(srcimg);
		boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
		for (int i = 0; i < boxes.size(); i++) {
			PaddleOCR::OCRPredictResult res;
			res.box = boxes[i];
			ocr_results.push_back(res);
		}
		// sort boex from top to bottom, from left to right
		PaddleOCR::Utility::sorted_boxes(ocr_results);
	}

	bool POV::PPOcrVinoRec::loadModel(const QString modelPath)
	{
		try
		{
			ov::Core core;
			auto model = core.read_model(modelPath.toStdString());
			ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
			infer_request = compiled_model.create_infer_request();
		}
		catch (const ov::Exception& e)
		{
			qDebug() << u8"错误：" << QString::fromStdString(e.default_msg) << e.what();
            return false;
		}
		return true;
	}

	void POV::PPOcrVinoRec::detect(std::vector<cv::Mat>& img_list, std::vector<std::string>& rec_texts_, std::vector<float>& rec_text_scores_)
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

			int batch_width = imgW;
			std::vector<cv::Mat> norm_img_batch;
			for (int ino = beg_img_no; ino < end_img_no; ino++) {
				cv::Mat srcimg;
				img_list[indices[ino]].copyTo(srcimg);
				cv::Mat resize_img;
				this->resize_op_.Run(srcimg, resize_img, max_wh_ratio,
					false, this->rec_image_shape_);
				this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
					true);
				norm_img_batch.push_back(resize_img);
				batch_width = std::max(resize_img.cols, batch_width);
			}

			std::vector<float> input(batch_num * 3 * imgH * batch_width, 0.0f);
			this->permute_op_.Run(norm_img_batch, input.data());
			// Inference.
			auto input_tensor = ov::Tensor(infer_request.get_input_tensor(0).get_element_type(),
				{ static_cast<unsigned long long>(batch_num), 3, static_cast<unsigned long long>(imgH), static_cast<unsigned long long>(batch_width) }, input.data());
			infer_request.set_input_tensor(input_tensor);
			infer_request.infer();

			ov::Tensor output_tensor = infer_request.get_output_tensor();
			const float* predict_batch = (float*)output_tensor.data();
			const ov::Shape predict_shape = output_tensor.get_shape();

			// ctc decode
			auto postprocess_start = std::chrono::steady_clock::now();
			for (int m = 0; m < predict_shape[0]; m++) {
				std::string str_res;
				int argmax_idx;
				int last_index = 0;
				float score = 0.f;
				int count = 0;
				float max_value = 0.0f;

				for (int n = 0; n < predict_shape[1]; n++) {
					// get idx
					argmax_idx = int(Utility::argmax(
						&predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
						&predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
					// get score
					max_value = float(*std::max_element(
						&predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
						&predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]]));

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
				rec_texts[indices[beg_img_no + m]] = str_res;
				rec_text_scores[indices[beg_img_no + m]] = score;
			}
			rec_texts_ = rec_texts;
			rec_text_scores_ = rec_text_scores;
		}
	}
	void POV::PPOcrVinoRec::Run(std::vector<PaddleOCR::OCRPredictResult> ocr_result, const cv::Mat& img,
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
	void POV::PPOcrVinoRec::Run(const cv::Mat& img, std::vector<std::string>& rec_texts_, std::vector<float>& rec_text_scores_)
	{
		std::vector<cv::Mat> img_list = { img };
		detect(img_list, rec_texts_, rec_text_scores_);
	}
	void POV::PPOcrVinoRec::Run(std::vector<cv::Mat>& img_list, std::vector<std::string>& rec_texts_, std::vector<float>& rec_text_scores_)
	{
		detect(img_list, rec_texts_, rec_text_scores_);
	}
}

