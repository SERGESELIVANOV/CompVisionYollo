#ifndef PHOTOPROCESSOR_H
#define PHOTOPROCESSOR_H

#include <QObject>
#include <QStringList>
#include <QImage>
#include <QFileInfo>

class PhotoProcessor : public QObject
{
    Q_OBJECT

public:
    explicit PhotoProcessor(const QStringList &photoPaths, QObject *parent = nullptr);

public slots:
    void processPhotos();

signals:
    void photoProcessed(int index, QString result);
    void finished();
    void errorOccurred(QString error);

private:
    QStringList m_photoPaths;

    QString analyzePhoto(const QString &filePath);
    QString mockAnalysis(const QImage &image, const QString &filePath);
};

#endif
