#pragma once

#include <QtWidgets/QMainWindow>
#include <QProcess>
#include <QFileDialog>
#include <QDir>
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
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onProcessError(QProcess::ProcessError error);
    void onReadyReadStandardOutput();
    void onReadyReadStandardError();

private:
    Ui::GUIClass ui;
    QProcess* process;

    void logMessage(const QString& message);
    void setControlsEnabled(bool enabled);
    QString getModelPreset() const;
};

