#pragma once

#include <QObject>
#include <QThread>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include "Libraries.h"
#include "Constans.h"
#include "Draw.h"
#include "pred_and_post_processes.h"
#include "creating_folders.h"

class Worker : public QObject
{
    Q_OBJECT

public:
    struct ModelPreset
    {
        std::string weights;
        std::string labels;
        cv::Size input_size{ 640, 640 };
        float conf_threshold{ 0.5f };
        float nms_threshold{ 0.45f };
    };

    Worker(QObject* parent = nullptr);
    ~Worker();

    void setParameters(const QString& modelType, const QString& inputDir, const QString& outputDir);

signals:
    void progressUpdated(int percentage);
    void logMessage(const QString& message);
    void processingFinished(bool success, const QString& message);
    void initializationFinished(bool success);

public slots:
    void process();

private:
    QString m_modelType;
    QString m_inputDir;
    QString m_outputDir;

    std::unordered_map<std::string, ModelPreset> m_presets;
    ModelPreset m_currentPreset;
    std::filesystem::path m_inputPath;
    std::filesystem::path m_outputPath;
    cv::dnn::Net m_net;
    std::vector<std::string> m_classList;
    std::vector<std::string> m_outputLayers;
    InferenceParams m_inferenceParams;

    bool initialize();
    bool isSupportedImage(const std::filesystem::path& path) const;
    bool loadClassList(const std::string& names_path, std::vector<std::string>& class_list) const;
};