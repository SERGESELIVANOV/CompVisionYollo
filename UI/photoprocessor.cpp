#include "photoprocessor.h"
#include <QFileInfo>
#include <QDebug>
#include <QElapsedTimer>
#include <QCoreApplication>
#include <QThread>

PhotoProcessor::PhotoProcessor(const QStringList &photoPaths, QObject *parent)
    : QObject(parent)
    , m_photoPaths(photoPaths)
{
}

void PhotoProcessor::processPhotos()
{
    QElapsedTimer timer;
    timer.start();

    for (int i = 0; i < m_photoPaths.count(); ++i) {
        QString result = analyzePhoto(m_photoPaths[i]);
        emit photoProcessed(i, result);

        // Обрабатываем события для обновления GUI
        QCoreApplication::processEvents();

        // Для демонстрации прогресса - небольшая задержка
        if (i % 50 == 0) {
            QThread::msleep(1);
        }
    }

    qDebug() << "Processing completed in" << timer.elapsed() << "ms";
    emit finished();
}

QString PhotoProcessor::analyzePhoto(const QString &filePath)
{
    QImage image(filePath);
    if (image.isNull()) {
        return "Ошибка: не удалось загрузить изображение";
    }

    return mockAnalysis(image, filePath);
}

QString PhotoProcessor::mockAnalysis(const QImage &image, const QString &filePath)
{
    QFileInfo info(filePath);

    // Пример анализа (замените на реальный)
    QStringList categories;

    // Анализ пропорций
    double ratio = (double)image.width() / image.height();
    if (ratio > 1.3) {
        categories << "Пейзаж";
    } else if (ratio < 0.8) {
        categories << "Портрет";
    } else {
        categories << "Квадрат";
    }

    // Анализ цветности
    if (image.isGrayscale()) {
        categories << "Черно-белое";
    } else {
        categories << "Цветное";

        // Простой анализ доминирующих цветов
        int colorCount = image.colorCount();
        if (colorCount > 0 && colorCount < 256) {
            categories << "Индексированное";
        }
    }

    // Анализ размера
    qint64 sizeKB = info.size() / 1024;
    QString sizeCategory;
    if (sizeKB < 100) {
        sizeCategory = "Маленькое";
    } else if (sizeKB < 1000) {
        sizeCategory = "Среднее";
    } else {
        sizeCategory = "Большое";
    }

    return QString("%1 | %2x%3 | %4 KB (%5) | Категории: %6")
        .arg(info.fileName())
        .arg(image.width())
        .arg(image.height())
        .arg(sizeKB)
        .arg(sizeCategory)
        .arg(categories.join(", "));
}
