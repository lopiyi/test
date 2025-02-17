#pragma once
#include <openvino/openvino.hpp>
#include <QString>
#include "OCR/include/postprocess_op.h"
#include "OCR/include/utility.h"
#include <OCR/include/preprocess_op.h>

using namespace PaddleOCR;
namespace POV
{
	class PPOcrVinoDet
	{
	public:
		PPOcrVinoDet() {}
		~PPOcrVinoDet() {}
		bool loadModel(const QString modelPath);
		std::vector<PaddleOCR::OCRPredictResult> Run(cv::Mat& img);
	private:
		void preprocess(const cv::Mat& img, cv::Mat& pred_map, cv::Mat& cbuf_map, float& ratio_h, float& ratio_w);
		void post_processor(const cv::Mat& img, std::vector<PaddleOCR::OCRPredictResult>& ocr_results,
			cv::Mat& pred_map, cv::Mat& bit_map, float& ratio_h, float& ratio_w);

		ov::InferRequest infer_request;
		PaddleOCR::DBPostProcessor post_processor_;
		int limit_side_len = 960;
		double det_db_thresh_ = 0.3;
		double det_db_box_thresh_ = 0.5;
		double det_db_unclip_ratio_ = 1.5;
		std::string det_db_score_mode_ = "slow";
	};

	class PPOcrVinoRec
	{
	public:
		PPOcrVinoRec(const std::string& label_path) {
			this->label_list_ = Utility::ReadDict(label_path);
			this->label_list_.insert(this->label_list_.begin(),
				"#"); // blank char for ctc
			this->label_list_.push_back(" ");
		}
		bool loadModel(const QString fileName);
		void Run(std::vector<PaddleOCR::OCRPredictResult>ocr_results, const cv::Mat& img,
			std::vector<std::string>& rec_texts_,
			std::vector<float>& rec_text_scores_);
		void Run(const cv::Mat& img,
			std::vector<std::string>& rec_texts_,
			std::vector<float>& rec_text_scores_);
		void Run(std::vector<cv::Mat>& img_list,
			std::vector<std::string>& rec_texts_,
			std::vector<float>& rec_text_scores_);
	private:
		void detect(std::vector<cv::Mat>& img_list,
			std::vector<std::string>& rec_texts_,
			std::vector<float>& rec_text_scores_);

		CrnnResizeImg resize_op_;
		Normalize normalize_op_;
		PermuteBatch permute_op_;

		ov::InferRequest infer_request;

		std::vector<float> mean_ = { 0.5f, 0.5f, 0.5f };
		std::vector<float> scale_ = { 1 / 0.5f, 1 / 0.5f, 1 / 0.5f };
		int rec_batch_num_ = 6;
		int rec_img_h_ = 48;
		int rec_img_w_ = 320;
		std::vector<int> rec_image_shape_ = { 3, rec_img_h_, rec_img_w_ };

		std::vector<std::string> label_list_;
	};
};

