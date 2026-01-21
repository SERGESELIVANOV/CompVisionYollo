<<<<<<< Current (Your changes)
=======
#pragma once

#include <QtWidgets/QMainWindow>
#include <QFileDialog>
#include <QDir>
#include <QFutureWatcher>
#include <QPointer>
#include "ComputerVision.h"
#include "ui_GUI.h"

class GUI : public QMainWindow
{
    Q_OBJECT

public:
    GUI(QWidget *parent = nullptr);
    ~GUI();

private slots:
    void on_inputDirButton_clicked();
    void on_outputDirButton_clicked();
    void on_startButton_clicked();
    void onProcessingFinished();

private:
    Ui::GUIClass ui;
    QFutureWatcher<int> m_futureWatcher;
    bool m_isProcessing{ false };

    void logMessage(const QString& message);
    void setControlsEnabled(bool enabled);
    QString getModelPreset() const;
};

>>>>>>> Incoming (Background Agent changes)
