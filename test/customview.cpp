#include "customview.h"
#include <QWheelEvent>
#include <QMouseEvent>
#include <QApplication>
#include <QGraphicsPixmapItem>
#include <QFileDialog>
#include <QPainter>
#include <QScrollBar>
#include <QDebug>

CustomView::CustomView(QWidget *parent)
	: QGraphicsView(parent), currentPixmapItem(nullptr), isDragging(false), 
	rubberBandMode(false), rubberBandItem(nullptr)
{
    setRenderHint(QPainter::Antialiasing);
    setDragMode(QGraphicsView::NoDrag);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
}

CustomView::~CustomView()
{}

void CustomView::setCurrentPixmapItem(QGraphicsPixmapItem* item)
{
    currentPixmapItem = item;
}

void CustomView::wheelEvent(QWheelEvent* event)
{
    if (event->modifiers() & Qt::ControlModifier) return;

    double scaleFactor = (event->angleDelta().y() > 0) ? 1.1 : 0.9;
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    scale(scaleFactor, scaleFactor);
}

void CustomView::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        if (QApplication::keyboardModifiers() & Qt::ControlModifier) {
            rubberBandStartPos = event->pos();
            rubberBandMode = true;

            QPointF scenePos = mapToScene(rubberBandStartPos);
            rubberBandItem = new QGraphicsRectItem();
            rubberBandItem->setPen(QPen(Qt::red, 1));
            scene()->addItem(rubberBandItem);
            rubberBandItem->setRect(QRectF(scenePos, QSizeF()));
        }
        else {
            isDragging = true;
            dragStartPos = event->pos();
            dragStartScrollPos = QPoint(horizontalScrollBar()->value(), verticalScrollBar()->value());
            setCursor(Qt::ClosedHandCursor);
        }
    }
    QGraphicsView::mousePressEvent(event);
}

void CustomView::mouseMoveEvent(QMouseEvent* event)
{
    if (rubberBandMode) {
        QPointF currentScenePos = mapToScene(event->pos());
        QRectF rect(mapToScene(rubberBandStartPos), currentScenePos);
        rubberBandItem->setRect(rect.normalized());
    }
    else if (isDragging) {
        QPoint delta = event->pos() - dragStartPos;
        horizontalScrollBar()->setValue(dragStartScrollPos.x() - delta.x());
        verticalScrollBar()->setValue(dragStartScrollPos.y() - delta.y());
    }
    QGraphicsView::mouseMoveEvent(event);
}

void CustomView::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        if (rubberBandMode) {
            rubberBandMode = false;
            if (rubberBandItem) {
                QRectF sceneRect = rubberBandItem->rect();
                scene()->removeItem(rubberBandItem);
                delete rubberBandItem;
                rubberBandItem = nullptr;

                if (sceneRect.isValid() && currentPixmapItem) {
                    QRectF itemRect = currentPixmapItem->mapFromScene(sceneRect).boundingRect();
                    QRect pixelRect = itemRect.toRect().intersected(currentPixmapItem->pixmap().rect());
                    if (!pixelRect.isEmpty()) {
                        QPixmap cropped = currentPixmapItem->pixmap().copy(pixelRect);
                        auto cvMat = QPixmapToCvMat(cropped);
                        std::vector<std::string> rec_texts;
                        std::vector<float> rec_text_scores;
                        std::vector<PaddleOCR::OCRPredictResult> result;
                        
                        if (det == nullptr)
                        {
                            #if CPU
                            det = new POV::PPOcrVinoDet;
                            det->loadModel("models/det.sim.xml");
                            rec = new POV::PPOcrVinoRec("models/ppocr_keys_v1.txt");
                            rec->loadModel("models/rec.sim.xml");
                            #else
                            det = new POT::PPOcrTrtDet;
                            det->loadModel("models/det.sim.trt");
                            rec = new POT::PPOcrTrtRec("models/ppocr_keys_v1.txt");
                            rec->loadModel("models/rec.sim.trt");
                            #endif
                        }
                        

                        if (use_det)
                        {
                            result = det->Run(cvMat);
                            rec->Run(result, cvMat, rec_texts, rec_text_scores);
                        }
                        else {
                            rec->Run(cvMat, rec_texts, rec_text_scores);
                        }
                        //Utility::VisualizeBboxes(cvMat, result,"1.jpg");
                        QList<OCRResult> results = getOCRResults(result, rec_texts, pixelRect);
                        showOCRResults(results);
                        //QString filename = QFileDialog::getSaveFileName(this, u8"保存截图", "", "PNG图像 (*.png);;JPEG图像 (*.jpg)");
                        //if (!filename.isEmpty()) {
                        //    cropped.save(filename);
                        //}
                    }
                }
            }
        }
        else if (isDragging) {
            isDragging = false;
            setCursor(Qt::ArrowCursor);
        }
    }
    QGraphicsView::mouseReleaseEvent(event);
}

cv::Mat CustomView::QPixmapToCvMat(const QPixmap& pixmap)
{
    QImage image = pixmap.toImage();
    if (image.isNull()) {
        return cv::Mat();
    }
    cv::Mat mat(image.height(), image.width(), CV_8UC4, const_cast<uchar*>(image.bits()), image.bytesPerLine());
    cv::Mat matBGR;
    cv::cvtColor(mat, matBGR, cv::COLOR_RGBA2BGR);

    return matBGR;
}

void CustomView::clearOCRResults() {
    foreach(QGraphicsItemGroup * group, ocrItems) {
        this->scene()->removeItem(group);
        delete group;
    }
    ocrItems.clear();
}

void CustomView::showOCRResults(const QList<OCRResult>& results) {
    clearOCRResults(); // 先清除旧结果

    foreach(const OCRResult & result, results) {
        // 创建图形组
        QGraphicsItemGroup* group = new QGraphicsItemGroup();
        scene()->addItem(group);

        // 绘制四边形
        QPolygonF polygon;
        foreach(const QPointF & point, result.points) {
            polygon << currentPixmapItem->mapToScene(point);
        }
        QGraphicsPolygonItem* polyItem = new QGraphicsPolygonItem(polygon);
        polyItem->setPen(QPen(Qt::green, 1));
        group->addToGroup(polyItem);

        // 添加文字
        QGraphicsTextItem* textItem = new QGraphicsTextItem(result.text);
        textItem->setPos(polygon.boundingRect().topRight() /*+ QPointF(10, 0)*/);
        textItem->setDefaultTextColor(Qt::red);
        textItem->setZValue(1); // 确保文字在顶层
        group->addToGroup(textItem);

        ocrItems.append(group);
    }
}

QList<CustomView::OCRResult> CustomView::getOCRResults(std::vector<PaddleOCR::OCRPredictResult>& ocr_result, std::vector<std::string>& rec_texts, QRect rect)
{
    QList<OCRResult> results;
    if (use_det)
    {
        for (int n = 0; n < ocr_result.size(); n++) {
            OCRResult result;
            QPolygonF polygon;
            for (int m = 0; m < ocr_result[n].box.size(); m++) {
                polygon << QPoint(int(ocr_result[n].box[m][0]), int(ocr_result[n].box[m][1])) + rect.topLeft();
            }
            result.points = polygon;
            result.text = QString::fromLocal8Bit(rec_texts[n].c_str());
            //qDebug() << result.text;
            results << result;
        }
    }
    else {
        for (int n = 0; n < rec_texts.size(); n++) {
            OCRResult result;
            QPolygonF polygon;
            polygon<<rect.topLeft()<<rect.topRight()<<rect.bottomRight()<<rect.bottomLeft();
            result.points = polygon;
            result.text = QString::fromLocal8Bit(rec_texts[n].c_str());
            //qDebug() << result.text;
            results << result;
        }
    }
    return results;
}