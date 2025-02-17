#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_test.h"
#include "customview.h"
#include <QPushButton>

class test : public QMainWindow
{
    Q_OBJECT

public:
    test(QWidget *parent = nullptr);
    ~test();
private slots:
    void onLoadButtonClicked();
private:
    Ui::testClass ui;

    QGraphicsScene* scene;
    CustomView* view;
    QPushButton* loadButton;
    QGraphicsPixmapItem* currentPixmapItem;
};
