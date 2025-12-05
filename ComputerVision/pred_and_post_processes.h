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
    // Предварительный расчет
    const float x_factor = static_cast<float>(input_image.cols) / INPUT_WIDTH;
    const float y_factor = static_cast<float>(input_image.rows) / INPUT_HEIGHT;
    // Указатель на данные один раз
    const float* data = outputs[0].ptr<float>();
    const int dimensions = class_name.size() + 5;
    const int rows = 25200;
    // Расчет порогов
    const float confidence_threshold = CONFIDENCE_THRESHOLD;
    const float score_threshold = SCORE_THRESHOLD;
    // Резервируем память (Решил сделать 10 процентов обнаружений)
    result.class_ids.reserve(rows / 10);
    result.confidences.reserve(rows / 10);
    result.boxes.reserve(rows / 10);
    result.class_names.reserve(rows / 10);
    // Цикл обработки
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence >= confidence_threshold)
        {
            const float* classes_scores = data + 5;
            // Ручной поиск максимального класса (быстрее чем minMaxLoc)
            int max_class_id = 0;
            float max_class_score = classes_scores[0];
            for (int j = 1; j < class_name.size(); ++j)
            {
                if (classes_scores[j] > max_class_score)
                {
                    max_class_score = classes_scores[j];
                    max_class_id = j;
                }
            }

            if (max_class_score > score_threshold)
            {
                result.confidences.push_back(confidence);
                result.class_ids.push_back(max_class_id);
                // Вычисление координат огр рамки
                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];

                int left = static_cast<int>((cx - 0.5f * w) * x_factor);
                int top = static_cast<int>((cy - 0.5f * h) * y_factor);
                int width = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);
                result.boxes.push_back(cv::Rect(left, top, width, height));
                result.class_names.push_back(class_name[max_class_id]);
            }
        }
        data += dimensions;
    }
    return result;
}