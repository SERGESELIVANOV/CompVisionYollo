#include "ObjectDetector.h"
#include <QDir>
#include <QFile>
#include <QStandardPaths>

ObjectDetector::ObjectDetector(QObject* parent)
    : QObject(parent)
    , m_totalImages(0)
    , m_processedImages(0)
{
    // Настройки путей по умолчанию
    const QString basePath = QDir::currentPath() + "/../Materials";

    // Предустановленные конфигурации моделей YOLO
    m_presets = {
        { "yolo11", { (basePath + "/yolo11n.onnx").toStdString(),
                      (basePath + "/coco.names").toStdString(),
                      cv::Size(640, 640), 0.5f, 0.45f } },
        { "yolov5", { (basePath + "/best.onnx").toStdString(),
                      (basePath + "/lvis.names").toStdString(),
                      cv::Size(640, 640), 0.4f, 0.45f } }
    };
}

ObjectDetector::~ObjectDetector()
{
}

bool ObjectDetector::initialize(const QString& modelType, const QString& inputDir, const QString& outputDir)
{
    auto preset_it = m_presets.find(modelType.toStdString());
    if (preset_it == m_presets.end())
    {
        emit logMessage("Неизвестный preset: " + modelType);
        return false;
    }

    m_currentPreset = preset_it->second;
    m_inputPath = std::filesystem::path(inputDir.toStdString());
    m_outputPath = std::filesystem::path(outputDir.toStdString());

    // Проверяем существование файлов и папок
    if (!std::filesystem::exists(m_currentPreset.weights))
    {
        emit logMessage("Ошибка открытия весов: " + QString::fromStdString(m_currentPreset.weights));
        return false;
    }

    if (!std::filesystem::exists(m_currentPreset.labels))
    {
        emit logMessage("Ошибка открытия файла меток: " + QString::fromStdString(m_currentPreset.labels));
        return false;
    }

    if (!std::filesystem::exists(m_inputPath))
    {
        emit logMessage("Ошибка доступа к папке с изображениями: " + inputDir);
        return false;
    }

    std::error_code out_ec;
    std::filesystem::create_directories(m_outputPath, out_ec);
    if (!std::filesystem::exists(m_outputPath))
    {
        emit logMessage("Ошибка доступа к папке для сохранения: " + outputDir);
        return false;
    }

    // Загружаем список классов
    if (!loadClassList(m_currentPreset.labels, m_classList))
    {
        emit logMessage("Не удалось загрузить список классов.");
        return false;
    }

    // Загружаем модель
    m_net = cv::dnn::readNet(m_currentPreset.weights);
    if (m_net.empty())
    {
        emit logMessage("Не удалось загрузить модель: " + QString::fromStdString(m_currentPreset.weights));
        return false;
    }

    // Получаем имена выходных слоев сети
    m_outputLayers = m_net.getUnconnectedOutLayersNames();
    m_inferenceParams = { m_currentPreset.input_size, m_currentPreset.conf_threshold, m_currentPreset.nms_threshold };

    emit logMessage("Модель успешно инициализирована: " + modelType);
    return true;
}

void ObjectDetector::processImages()
{
    m_processedImages = 0;
    m_totalImages = 0;

    // Подсчитываем общее количество изображений
    for (const auto& entry : std::filesystem::directory_iterator(m_inputPath))
    {
        if (entry.is_regular_file() && isSupportedImage(entry.path()))
        {
            m_totalImages++;
        }
    }

    if (m_totalImages == 0)
    {
        emit processingFinished(false, "В указанной папке не найдено подходящих изображений");
        return;
    }

    emit logMessage(QString("Найдено изображений для обработки: %1").arg(m_totalImages));
    emit progressUpdated(0);

    // Обрабатываем все изображения
    for (const auto& entry : std::filesystem::directory_iterator(m_inputPath))
    {
        if (!entry.is_regular_file() || !isSupportedImage(entry.path()))
        {
            continue;
        }

        cv::Mat frame = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (frame.empty())
        {
            emit logMessage("Ошибка чтения изображения: " + QString::fromStdWString(entry.path().wstring()));
            continue;
        }

        // Препроцессинг: подготовка изображения и запуск нейросети
        std::vector<cv::Mat> detections = pre_process(frame, m_net, m_inferenceParams, m_outputLayers);
        // Постпроцессинг: парсинг выходов сети, фильтрация по порогам, NMS
        Detection detection_photo = post_process(frame, detections, m_classList, m_inferenceParams);
        // Отрисовка найденных объектов на изображении
        cv::Mat img = drawDetections(frame, detection_photo.boxes, detection_photo.confidences, detection_photo.class_ids, m_classList);

        // Измерение времени инференса и отрисовка на изображении
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000.0;
        double t = m_net.getPerfProfile(layersTimes) / freq;
        char time_label[64];
        snprintf(time_label, sizeof(time_label), "Inference time: %.2f ms", t);
        cv::putText(img, time_label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, cv::Scalar(0, 0, 255));

        // Сохранение: берем первый найденный класс или "no_detection" если ничего не найдено
        const std::string main_tag = detection_photo.class_names.empty() ? "no_detection" : detection_photo.class_names.front();
        saveProcessedImage(img, main_tag, m_outputPath);

        m_processedImages++;
        int progress = static_cast<int>((m_processedImages * 100) / m_totalImages);
        emit progressUpdated(progress);

        emit logMessage(QString("Обработано: %1/%2").arg(m_processedImages).arg(m_totalImages));
    }

    emit processingFinished(true, QString("Обработано изображений: %1").arg(m_processedImages));
}

bool ObjectDetector::isSupportedImage(const std::filesystem::path& path) const
{
    if (!path.has_extension())
    {
        return false;
    }

    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

bool ObjectDetector::loadClassList(const std::string& names_path, std::vector<std::string>& class_list) const
{
    std::ifstream ifs(names_path);
    if (!ifs.is_open())
    {
        return false;
    }

    std::string line;
    while (std::getline(ifs, line))
    {
        if (!line.empty())
        {
            class_list.push_back(line);
        }
    }
    return !class_list.empty();
}