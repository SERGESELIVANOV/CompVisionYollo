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
    cv::Mat result_image = input_image.clone();
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        if (idx < 0 || idx >= boxes.size() || idx >= confidences.size() ||
            idx >= class_ids.size() || class_ids[idx] < 0 || class_ids[idx] >= class_name.size()) {
            std::wcout << L"Неверный индекс или class_id: " << idx
                << L", class_id: " << (idx < class_ids.size() ? class_ids[idx] : -1)
                << L", размер class_name: " << class_name.size() << std::endl;
            continue;
        }
        cv::Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // рисовка ограничивающую рамку 
        cv::rectangle(result_image, cv::Point(left, top), cv::Point(left + width, top + height), BLUE, 3 * THICKNESS);
        // Получаем метку для имени класса и его достоверности 
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << confidences[idx];
        std::string label = class_name[class_ids[idx]] + ":" + ss.str();
        // Нарисуйте метки классов.
        draw_label(result_image, label, left, top);
    }
    return result_image;
}