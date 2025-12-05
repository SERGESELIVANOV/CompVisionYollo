#pragma once
//Функция добавления анатаций к названиям класса 
void draw_label(cv::Mat& input_image, std::string label, int left, int top)
{
    // Отображение тегга в верхней части ограничивающего прямоугольника 
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = std::max(top, label_size.height);
    // Верхний левый угол 
    cv::Point tlc = cv::Point(left, top);
    // Нижний правый угол 
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
    // Рисовка белого прямоугольника 
    cv::rectangle(input_image, tlc, brc, BLACK, cv::FILLED);
    // Рисовка тэга на прямоугольник 
    cv::putText(input_image, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

cv::Mat drawDetections(cv::Mat& input_image, const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, const std::vector<int>& class_ids, const std::vector<std::string>& class_name)
{
    std::vector<int> indices;
    cv::Mat result_image;

    // экономит время на ненужном вызове NMSBoxes и клонировании
    if (boxes.empty())
    {
        // копируем вместо клонирования - быстрее для пустых случаев
        input_image.copyTo(result_image);
        return result_image;  //экономит много времени для пустых изображений
    }

    // предотвращаем многократный доступ к глобальным переменным
    const float score_threshold = SCORE_THRESHOLD;
    const float nms_threshold = NMS_THRESHOLD;

    cv::dnn::NMSBoxes(boxes, confidences, score_threshold, nms_threshold, indices);

    // проверка результата NMS перед клонированием, т.к многие изображения могут не иметь детекций после NMS
    if (indices.empty())
    {
        input_image.copyTo(result_image);
        return result_image;
    }

    // только теперь клонируем изображение - когда уверены что будем рисовать
    result_image = input_image.clone();
    const int thickness = 3 * THICKNESS;
    const cv::Scalar color = BLUE;
    const size_t indices_size = indices.size();

    for (size_t i = 0; i < indices_size; ++i)
    {
        int idx = indices[i];
        if (static_cast<size_t>(idx) >= boxes.size() ||
            static_cast<size_t>(idx) >= confidences.size() ||
            static_cast<size_t>(idx) >= class_ids.size())
        {
            continue;
        }

        int class_id = class_ids[idx];
        // избегаем повторного доступа class_ids[idx]
        if (class_id < 0 || static_cast<size_t>(class_id) >= class_name.size())
        {
            continue;
        }
        const cv::Rect& box = boxes[idx];

        // вычисляем все координаты один раз
        int left = box.x;
        int top = box.y;
        int right = left + box.width;
        int bottom = top + box.height;
        cv::rectangle(result_image, cv::Point(left, top), cv::Point(right, bottom),
            color, thickness);

        // статический буфер
        float confidence = confidences[idx];
        char confidence_str[16];

        //snprintf не использует динамическое выделение памяти (вместо stringstream)
        snprintf(confidence_str, sizeof(confidence_str), "%.2f", confidence);
        std::string label = class_name[class_id];
        label += ":";
        label += confidence_str;
        draw_label(result_image, label, left, top);
    }
    return result_image;
}