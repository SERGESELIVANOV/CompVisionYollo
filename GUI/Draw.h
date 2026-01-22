#pragma once
#include "Constans.h"

// Рисует подпись с классом и уверенностью над рамкой детекции
inline void draw_label(cv::Mat& input_image, const std::string& label, int left, int top)
{
    // Вычисляем размер текста для корректного размещения
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = std::max(top, label_size.height);  // Поднимаем подпись, если рамка слишком высоко
    // Координаты прямоугольника-подложки для текста
    cv::Point tlc = cv::Point(left, top);
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
    cv::rectangle(input_image, tlc, brc, getBlackColor(), cv::FILLED);  // Черная подложка
    cv::putText(input_image, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, getYellowColor(), THICKNESS);
}

// Отрисовывает все детекции на изображении: рамки и подписи с классами
inline cv::Mat drawDetections(cv::Mat& input_image, const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, const std::vector<int>& class_ids, const std::vector<std::string>& class_name)
{
    cv::Mat result_image;
    input_image.copyTo(result_image);
    if (boxes.empty())
    {
        return result_image;
    }

    const int thickness = 3 * THICKNESS;
    const cv::Scalar color = getBlueColor();
    // Безопасное определение количества детекций (на случай несоответствия размеров векторов)
    const size_t detections = std::min({ boxes.size(), confidences.size(), class_ids.size() });

    for (size_t i = 0; i < detections; ++i)
    {
        const cv::Rect& box = boxes[i];
        if (!box.area())  // Пропускаем пустые рамки
        {
            continue;
        }

        int class_id = class_ids[i];
        if (class_id < 0 || static_cast<size_t>(class_id) >= class_name.size())  // Проверка корректности ID
        {
            continue;
        }

        // Отрисовка рамки вокруг объекта
        int left = box.x;
        int top = box.y;
        int right = left + box.width;
        int bottom = top + box.height;
        cv::rectangle(result_image, cv::Point(left, top), cv::Point(right, bottom),
            color, thickness);

        // Формирование подписи: "класс: уверенность"
        float confidence = confidences[i];
        char confidence_str[16];
        snprintf(confidence_str, sizeof(confidence_str), "%.2f", confidence);  // Форматирование до 2 знаков
        std::string label = class_name[class_id];
        label += ":";
        label += confidence_str;
        draw_label(result_image, label, left, top);
    }
    return result_image;
}