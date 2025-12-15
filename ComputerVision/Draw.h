#pragma once
//Функция рисования подписи у рамки 
void draw_label(cv::Mat& input_image, const std::string& label, int left, int top)
{
    // Считаем размер текста для корректного размещения 
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = std::max(top, label_size.height);
    // Верхний левый угол подложки 
    cv::Point tlc = cv::Point(left, top);
    // Нижний правый угол подложки 
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
    // Рисуем прямоугольник-подложку 
    cv::rectangle(input_image, tlc, brc, BLACK, cv::FILLED);
    // Рисуем текст на подложке 
    cv::putText(input_image, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

cv::Mat drawDetections(cv::Mat& input_image, const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, const std::vector<int>& class_ids, const std::vector<std::string>& class_name)
{
    cv::Mat result_image;
    input_image.copyTo(result_image);
    if (boxes.empty())
    {
        return result_image;
    }

    const int thickness = 3 * THICKNESS;
    const cv::Scalar color = BLUE;
    const size_t detections = std::min({ boxes.size(), confidences.size(), class_ids.size() });

    for (size_t i = 0; i < detections; ++i)
    {
        const cv::Rect& box = boxes[i];
        if (!box.area())
        {
            continue;
        }

        int class_id = class_ids[i];
        if (class_id < 0 || static_cast<size_t>(class_id) >= class_name.size())
        {
            continue;
        }

        // Границы рамки
        int left = box.x;
        int top = box.y;
        int right = left + box.width;
        int bottom = top + box.height;
        cv::rectangle(result_image, cv::Point(left, top), cv::Point(right, bottom),
            color, thickness);

        // Подготовка подписи
        float confidence = confidences[i];
        char confidence_str[16];

        // snprintf экономичнее по памяти, чем stringstream
        snprintf(confidence_str, sizeof(confidence_str), "%.2f", confidence);
        std::string label = class_name[class_id];
        label += ":";
        label += confidence_str;
        draw_label(result_image, label, left, top);
    }
    return result_image;
}