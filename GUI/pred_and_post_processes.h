#pragma once
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <vector>
#include <string>

// Результат детекции: координаты рамок, уверенность, ID и имена классов
struct Detection
{
    std::vector<cv::Rect> boxes;           // Координаты ограничивающих рамок
    std::vector<float> confidences;        // Уверенность для каждой детекции (0.0-1.0)
    std::vector<int> class_ids;            // ID класса для каждой детекции
    std::vector<std::string> class_names;  // Имена классов (из .names файла)
};

// Параметры инференса: размер входа и пороги фильтрации
struct InferenceParams
{
    cv::Size input_size{ 640, 640 };
    float conf_threshold{ 0.5f };
    float nms_threshold{ 0.45f };
};

namespace
{
    // Структура для описания формата выхода модели (количество детекций и признаков)
    struct OutputLayout
    {
        int detections = 0;
        int features = 0;         // Количество признаков на детекцию (4 bbox + objectness + классы)
        bool features_last = true;  // true если признаки в последнем измерении [N, features]
    };

    // Анализирует размерности выхода модели и определяет формат данных
    inline OutputLayout analyzeOutputLayout(const cv::Mat& output, int expected_with_obj, int expected_without_obj)
    {
        OutputLayout layout{};
        const int dims = output.dims;
        std::vector<int> sizes(dims);
        for (int i = 0; i < dims; ++i)
        {
            sizes[i] = output.size[i];
        }

        layout.features = sizes.back();
        layout.features_last = true;

        for (int i = dims - 1; i >= 0; --i)
        {
            if (sizes[i] == expected_with_obj || sizes[i] == expected_without_obj)
            {
                layout.features = sizes[i];
                layout.features_last = (i == dims - 1);
                break;
            }
        }

        layout.detections = static_cast<int>(output.total() / std::max(layout.features, 1));
        return layout;
    }

    // Преобразует выход модели в матрицу [детекции x признаки] независимо от исходного формата
    inline cv::Mat flattenDetections(const cv::Mat& output, int features, bool features_last)
    {
        const int detections = static_cast<int>(output.total() / features);
        cv::Mat result(detections, features, CV_32F);
        const float* src = output.ptr<float>();
        float* dst = result.ptr<float>();
        // Если признаки уже в последнем измерении, просто копируем
        if (features_last)
        {
            std::memcpy(dst, src, static_cast<size_t>(detections) * features * sizeof(float));
            return result;
        }
        // Иначе транспонируем: [признаки x детекции] -> [детекции x признаки]
        for (int det = 0; det < detections; ++det)
        {
            for (int feat = 0; feat < features; ++feat)
            {
                dst[det * features + feat] = src[feat * detections + det];
            }
        }
        return result;
    }
}

// Препроцессинг: нормализация изображения, масштабирование и запуск сети
inline std::vector<cv::Mat> pre_process(cv::Mat& input_image, cv::dnn::Net& net,
    const InferenceParams& params, const std::vector<std::string>& output_layers)
{
    cv::Mat blob;
    // Преобразование: нормализация [0,255] -> [0,1], масштабирование до input_size, BGR->RGB
    cv::dnn::blobFromImage(input_image, blob, 1.f / 255.f, params.input_size, cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    outputs.reserve(output_layers.size());
    net.forward(outputs, output_layers);  // Запуск инференса
    return outputs;
}


// Постпроцессинг: парсинг выходов сети, фильтрация по порогам, NMS, масштабирование координат
inline Detection post_process(cv::Mat& input_image, std::vector<cv::Mat>& outputs,
    const std::vector<std::string>& class_name, const InferenceParams& params)
{
    Detection result;
    if (outputs.empty())
    {
        return result;
    }

    // Коэффициенты масштабирования для перевода координат из input_size в размер исходного изображения
    const float x_factor = static_cast<float>(input_image.cols) / params.input_size.width;
    const float y_factor = static_cast<float>(input_image.rows) / params.input_size.height;
    std::vector<cv::Rect> all_boxes;
    std::vector<float> all_confidences;
    std::vector<int> all_class_ids;

    const int class_count = static_cast<int>(class_name.size());
    // Ожидаемое количество признаков: с objectness (5) или без (4) + количество классов
    const int expected_with_obj = class_count + 5;      // YOLOv5: [cx, cy, w, h, obj, class1, ..., classN]
    const int expected_without_obj = class_count + 4;   // YOLO11: [cx, cy, w, h, class1, ..., classN]

    for (auto& output : outputs)
    {
        if (output.empty())
        {
            continue;
        }

        const OutputLayout layout = analyzeOutputLayout(output, expected_with_obj, expected_without_obj);
        if (layout.features <= 4 || layout.detections == 0)
        {
            continue;
        }

        cv::Mat detections = flattenDetections(output, layout.features, layout.features_last);
        // Определяем формат: YOLOv5 имеет objectness (5 признаков), YOLO11 - нет (4 признака)
        const bool has_objectness = (layout.features - class_count) == 5;
        const int class_offset = has_objectness ? 5 : 4;  // Смещение до начала классовых вероятностей
        if (layout.features < class_offset || layout.features < class_offset + class_count)
        {
            continue;
        }

        // Обработка каждой детекции
        for (int i = 0; i < detections.rows; ++i)
        {
            const float* detection = detections.ptr<float>(i);
            // Координаты в формате center-x, center-y, width, height (нормализованные)
            float cx = detection[0];
            float cy = detection[1];
            float width = detection[2];
            float height = detection[3];

            // Objectness (если есть) или 1.0 для моделей без objectness
            float object_conf = has_objectness ? detection[4] : 1.f;
            const float* class_scores = detection + class_offset;
            // Находим класс с максимальной вероятностью
            int best_class_id = -1;
            float best_score = 0.f;
            for (int c = 0; c < class_count; ++c)
            {
                const float score = class_scores[c];
                if (score > best_score)
                {
                    best_score = score;
                    best_class_id = c;
                }
            }

            if (best_class_id < 0)
            {
                continue;
            }

            // Итоговая уверенность = objectness * классовая вероятность
            const float confidence = object_conf * best_score;
            if (confidence < params.conf_threshold)
            {
                continue;
            }

            // Преобразование из center-формата в left-top-width-height и масштабирование
            int left = static_cast<int>((cx - width * 0.5f) * x_factor);
            int top = static_cast<int>((cy - height * 0.5f) * y_factor);
            int w = static_cast<int>(width * x_factor);
            int h = static_cast<int>(height * y_factor);

            // Ограничение координат границами изображения
            left = std::clamp(left, 0, std::max(input_image.cols - 1, 0));
            top = std::clamp(top, 0, std::max(input_image.rows - 1, 0));
            w = std::clamp(w, 0, input_image.cols - left);
            h = std::clamp(h, 0, input_image.rows - top);
            if (w <= 2 || h <= 2)  // Отбрасываем слишком маленькие рамки
            {
                continue;
            }

            all_boxes.emplace_back(left, top, w, h);
            all_confidences.push_back(confidence);
            all_class_ids.push_back(best_class_id);
        }
    }

    if (all_boxes.empty())
    {
        return result;
    }

    // Non-Maximum Suppression: удаление перекрывающихся детекций одного объекта
    std::vector<int> indices;
    cv::dnn::NMSBoxes(all_boxes, all_confidences, params.conf_threshold, params.nms_threshold, indices);
    // Сохраняем только детекции, прошедшие NMS
    for (int idx : indices)
    {
        result.boxes.push_back(all_boxes[idx]);
        result.confidences.push_back(all_confidences[idx]);
        result.class_ids.push_back(all_class_ids[idx]);
        // Получаем имя класса из списка или "unknown" если ID некорректный
        if (all_class_ids[idx] >= 0 && all_class_ids[idx] < class_count)
        {
            result.class_names.push_back(class_name[all_class_ids[idx]]);
        }
        else
        {
            result.class_names.emplace_back("unknown");
        }
    }

    return result;
}