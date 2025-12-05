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
    std::ifstream ifs("C:/detection/ComputerVision/lvis.names");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    const std::filesystem::path pathPhoto = "C:/Users/Сергей/Desktop/програмирование C++/Летний проект/photo";
    // Загрузка модели 
    cv::dnn::Net net;
    net = cv::dnn::readNet("C:/detection/ComputerVision/best.onnx");
    for (const auto& entry : std::filesystem::directory_iterator(pathPhoto))
    {
        const std::string extension = entry.path().extension().string();
        if (extension != ".jpg" && extension != ".jpeg" && extension != ".png" && extension != ".bmp")
        {
            continue;
        }

        cv::Mat frame = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (frame.empty())
        {
            std::wcout << L"Ошибка загрузки изображения: " << entry.path().wstring() << std::endl;
            continue;
        }
        std::vector<cv::Mat> detections = pre_process(frame, net);
        Detection detection_photo = post_process(frame, detections, class_list);
        cv::Mat img = drawDetections(frame, detection_photo.boxes, detection_photo.confidences, detection_photo.class_ids, class_list);
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000.0;  // 1000.0 вместо 1000
        double t = net.getPerfProfile(layersTimes) / freq;
        // snprintf вместо stringstream (stringstream из-за динамического выделения памяти использует избыточное кол-во ресурсов)
        char time_label[64];  // статический буфер
        snprintf(time_label, sizeof(time_label), "Inference time: %.2f ms", t);
        cv::putText(img, time_label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);
        std::string main_tag = detection_photo.class_names[0];
        // СОХРАНЯЕМ ФАЙЛ
        saveProcessedImage(img, main_tag);
    }
    return 0;
}