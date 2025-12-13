#pragma once
struct Detection
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<std::string> class_names;
};



std::vector<cv::Mat> pre_process(cv::Mat& input_image, cv::dnn::Net& net)
{
    cv::Mat blob;
    // Более быстрые параметры для blobFromImage
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);
    static std::vector<std::string> output_layers = net.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outputs;
    // Резерв памяти во избежание реаллокаций
    outputs.reserve(output_layers.size());
    net.forward(outputs, output_layers);
    return outputs;
}


Detection post_process(cv::Mat& input_image, std::vector<cv::Mat>& outputs, const std::vector<std::string>& class_name)
{
    Detection result;

    if (outputs.empty() || outputs[0].empty())
    {
        std::wcout << L" Выходы нейросети пустые" << std::endl;
        return result;
    }

    cv::Mat output = outputs[0];

    // ДЕБАГ: Выводим размеры тензора
    std::cout << "Размеры выхода: "
        << "dims=" << output.dims
        << ", size=[" << output.size[0] << ", "
        << output.size[1] << ", " << output.size[2] << "]"
        << std::endl;

    // Вариант 1: Если формат [1, 84, 8400]
    const int num_detections = output.size[2];  // 8400
    const int num_features = output.size[1];    // 84

    std::cout << "Особенности: " << num_features
        << ", Детекций: " << num_detections << std::endl;

    const float x_factor = static_cast<float>(input_image.cols) / INPUT_WIDTH;
    const float y_factor = static_cast<float>(input_image.rows) / INPUT_HEIGHT;

    const float confidence_threshold = CONFIDENCE_THRESHOLD;  // должно быть 0.5 или выше
    const float score_threshold = SCORE_THRESHOLD;            // должно быть 0.25 или выше

    // ВАЖНО: Используйте высокие пороги для фильтрации мусора
    const float FINAL_CONF_THRESHOLD = 0.5f;  // Минимум 50% уверенности
    const float FINAL_NMS_THRESHOLD = 0.45f;  // Порог для NMS

    // Ресайзим выход для удобства обработки
    cv::Mat detections(num_detections, num_features, CV_32F);

    // Копируем данные с транспонированием
    const float* src_data = output.ptr<float>();
    float* dst_data = detections.ptr<float>();

    // Транспонирование: [1, 84, 8400] -> [8400, 84]
    for (int i = 0; i < num_detections; ++i) {
        for (int j = 0; j < num_features; ++j) {
            dst_data[i * num_features + j] = src_data[j * num_detections + i];
        }
    }

    // Собираем детекции
    std::vector<cv::Rect> all_boxes;
    std::vector<float> all_confidences;
    std::vector<int> all_class_ids;

    for (int i = 0; i < num_detections; ++i) {
        const float* detection = detections.ptr<float>(i);

        // Первые 4 значения: cx, cy, width, height
        float cx = detection[0];
        float cy = detection[1];
        float width = detection[2];
        float height = detection[3];

        // Находим лучший класс (начиная с 4-го элемента)
        float best_class_confidence = 0;
        int best_class_id = -1;

        for (int c = 4; c < num_features; ++c) {
            float class_confidence = detection[c];
            if (class_confidence > best_class_confidence) {
                best_class_confidence = class_confidence;
                best_class_id = c - 4;  // -4 потому что первые 4 - bbox
            }
        }

        // Пропускаем если confidence слишком низкий
        if (best_class_confidence < FINAL_CONF_THRESHOLD) {
            continue;
        }

        // Преобразуем координаты
        int left = static_cast<int>((cx - width * 0.5f) * x_factor);
        int top = static_cast<int>((cy - height * 0.5f) * y_factor);
        int w = static_cast<int>(width * x_factor);
        int h = static_cast<int>(height * y_factor);

        // Проверяем валидность координат
        left = std::max(0, left);
        top = std::max(0, top);
        w = std::min(w, input_image.cols - left);
        h = std::min(h, input_image.rows - top);

        if (w <= 2 || h <= 2) continue;  // слишком маленький

        all_boxes.push_back(cv::Rect(left, top, w, h));
        all_confidences.push_back(best_class_confidence);
        all_class_ids.push_back(best_class_id);
    }

    std::cout << "После первичной фильтрации: " << all_boxes.size() << " детекций" << std::endl;

    // ПРИМЕНЯЕМ NMS для устранения дубликатов
    std::vector<int> indices;
    if (!all_boxes.empty()) {
        try {
            cv::dnn::NMSBoxes(all_boxes, all_confidences,
                FINAL_CONF_THRESHOLD, FINAL_NMS_THRESHOLD, indices);

            std::cout << "После NMS осталось: " << indices.size() << " детекций" << std::endl;
        }
        catch (const cv::Exception& e) {
            std::cerr << "Ошибка NMS: " << e.what() << std::endl;
            return result;
        }
    }

    // Заполняем результат
    for (int idx : indices) {
        result.boxes.push_back(all_boxes[idx]);
        result.confidences.push_back(all_confidences[idx]);
        result.class_ids.push_back(all_class_ids[idx]);

        if (all_class_ids[idx] >= 0 && all_class_ids[idx] < class_name.size()) {
            result.class_names.push_back(class_name[all_class_ids[idx]]);
        }
        else {
            result.class_names.push_back("unknown");
        }
    }

    // Выводим результаты
    std::cout << "Итоговых детекций: " << result.boxes.size() << std::endl;
    for (size_t i = 0; i < result.boxes.size() && i < 10; ++i) {
        std::cout << "  " << i << ": " << result.class_names[i]
            << " (conf: " << result.confidences[i]
            << ", box: [" << result.boxes[i].x << ", " << result.boxes[i].y
            << ", " << result.boxes[i].width << ", " << result.boxes[i].height
            << "])" << std::endl;
    }

    return result;
}