#pragma once

#include <QtWidgets/QMainWindow>
#include <QFileDialog>
#include <QDir>
#include <QThread>
#include "ui_GUI.h"
#include "Worker.h"

class GUI : public QMainWindow
{
    Q_OBJECT

public:
    GUI(QWidget* parent = nullptr);
    ~GUI();

private slots:
    void onInputDirButtonClicked();
    void onOutputDirButtonClicked();
    void onStartButtonClicked();
    void onProgressUpdated(int percentage);
    void onLogMessage(const QString& message);
    void onProcessingFinished(bool success, const QString& message);
    void onInitializationFinished(bool success);

private:
    Ui::GUIClass ui;
    Worker* m_worker;
    QThread* m_workerThread;
    bool m_isProcessing;

    void logMessage(const QString& message);
    void setControlsEnabled(bool enabled);
    QString getModelPreset() const;
};