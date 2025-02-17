#include "test.h"
#include <QVBoxLayout>
#include <QFileDialog>


test::test(QWidget *parent)
    : QMainWindow(parent), scene(new QGraphicsScene(this)), view(new CustomView), 
    loadButton(new QPushButton(u8"¼ÓÔØÍ¼Ïñ", this)), currentPixmapItem(nullptr)
{
    ui.setupUi(this);

    QWidget* centralWidget = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(centralWidget);
    layout->addWidget(loadButton);
    layout->addWidget(view);
    setCentralWidget(centralWidget);

    view->setScene(scene);
    connect(loadButton, &QPushButton::clicked, this, &test::onLoadButtonClicked);
}

test::~test()
{}

void test::onLoadButtonClicked()
{
    QString filename = QFileDialog::getOpenFileName(this, u8"´ò¿ªÍ¼Ïñ", "", "Images (*.png *.jpg *.bmp)");
    if (!filename.isEmpty()) {
        QPixmap pixmap(filename);
        if (!pixmap.isNull()) {
            scene->clear();
            currentPixmapItem = scene->addPixmap(pixmap);
            scene->setSceneRect(pixmap.rect());
            view->setCurrentPixmapItem(currentPixmapItem);
        }
    }
}