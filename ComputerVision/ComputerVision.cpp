#include "Libraries.h"
#include "Constans.h"
#include "Draw.h"
#include "pred_and_post_processes.h"
#include "creating_folders.h"


int main()
{
    setlocale(LC_ALL, "Russian");
    // Загрузить список классов
    std::vector<std::string> class_list;
    std::ifstream ifs("C:/Users/Сергей/Desktop/програмирование C++/Летний проект/lvis.names");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    const std::filesystem::path pathPhoto = "C:/Users/Сергей/Desktop/програмирование C++/Летний проект/photo";
    const std::string pathPhoto2 = "C:/Users/Сергей/Desktop/програмирование C++/Летний проект/";
    // Загрузка модели 
    cv::dnn::Net net;
    net = cv::dnn::readNet("C:/Users/Сергей/Desktop/програмирование C++/Летний проект/best.onnx");
    for (const auto& entry : std::filesystem::directory_iterator(pathPhoto))
    {
        cv::Mat frame;
        const auto file = entry.path();
        frame = cv::imread(file.string());
        std::vector<cv::Mat> detections;     // Обработка изображения 
        detections = pre_process(frame, net);
        cv::Mat frame_clone = frame.clone();
        Detection detection_photo = post_process(frame_clone, detections, class_list);
        cv::Mat img = drawDetections(frame, detection_photo.boxes, detection_photo.confidences, detection_photo.class_ids, class_list);
        // функция getPerfProfile для вывода работы.
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::stringstream time_ss;
        time_ss << std::fixed << std::setprecision(2) << "Inference time: " << t << " ms";
        std::string label = time_ss.str();
        cv::putText(img, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);
        std::string main_tag = detection_photo.class_names[0];
        // СОХРАНЯЕМ ФАЙЛ
        saveProcessedImage(img, main_tag);
    }
    return 0;
}