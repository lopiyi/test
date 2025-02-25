#pragma once

#include <QGraphicsView>
#include <QGraphicsRectItem>
#include <opencv2/opencv.hpp>
#include <QPixmap>
#include <QGraphicsItemGroup>

#define CPU false

#if CPU
#include "PPOcrVino.h"
#else
#include "PPOcrTrt.h"
#endif

class CustomView  : public QGraphicsView
{
	Q_OBJECT

public:
    struct OCRResult {
        QString text;
        QPolygonF points; // 四个点的四边形
    };
	CustomView(QWidget *parent=nullptr);
	~CustomView();
	void setCurrentPixmapItem(QGraphicsPixmapItem* item);
protected:
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
private:
    cv::Mat QPixmapToCvMat(const QPixmap& pixmap);
    void clearOCRResults(); 
    QList<OCRResult> getOCRResults(std::vector<PaddleOCR::OCRPredictResult>& ocr_result, std::vector<std::string>& rec_texts, QRect rect);
    void showOCRResults(const QList<OCRResult>& results);
    QList<QGraphicsItemGroup*> ocrItems; 
private:
    QGraphicsPixmapItem* currentPixmapItem;
    bool isDragging;
    QPoint dragStartPos;
    QPoint dragStartScrollPos;

    bool rubberBandMode;
    QPoint rubberBandStartPos;
    QGraphicsRectItem* rubberBandItem;

    #if CPU
    POV::PPOcrVinoDet* det = nullptr;
    POV::PPOcrVinoRec* rec = nullptr;
    #else
    POT::PPOcrTrtDet* det = nullptr;
    POT::PPOcrTrtRec* rec = nullptr;
    #endif

    bool use_det = true;
};
