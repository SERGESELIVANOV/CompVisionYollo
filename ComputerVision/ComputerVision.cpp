#include "Libraries.h"
#include "Constans.h"
#include "Draw.h"
#include "pred_and_post_processes.h"
#include "creating_folders.h"

namespace
{
    struct ModelPreset
    {
        std::string weights;
        std::string labels;
        cv::Size input_size{ 640, 640 };
        float conf_threshold{ 0.5f };
        float nms_threshold{ 0.45f };
    };

    bool loadClassList(const std::string& names_path, std::vector<std::string>& class_list)
    {
        std::ifstream ifs(names_path);
        if (!ifs.is_open())
        {
            std::cerr << "Ошибка открытия файла: " << names_path << std::endl;
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

    bool isSupportedImage(const std::filesystem::path& path)
    {
        if (!path.has_extension())
        {
            return false;
        }

        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
    }
}

int main()
{
    setlocale(LC_ALL, "Russian");

    // Настройки путей по умолчанию (можно менять прямо здесь)
    const std::string DEFAULT_IMAGES_DIR = "C:/Users/polezhaev/Desktop/Materials/photo";
    const std::string DEFAULT_OUTPUT_DIR = "C:/Users/polezhaev/source/repos/CompVisionYollo/ComputerVision/tags";
    const std::string DEFAULT_PRESET = "yolov5";

    const std::unordered_map<std::string, ModelPreset> presets = {
        { "yolo11", { "C:/Users/polezhaev/Desktop/Materials/yolo11n.onnx",
                      "C:/Users/polezhaev/Desktop/Materials/coco.names",
                      cv::Size(640, 640), 0.5f, 0.45f } },
        { "yolov5", { "C:/Users/polezhaev/Desktop/Materials/best.onnx",
                      "C:/Users/polezhaev/Desktop/Materials/lvis.names",
                      cv::Size(640, 640), 0.4f, 0.45f } }
    };

    auto preset_it = presets.find(DEFAULT_PRESET);
    if (preset_it == presets.end())
    {
        std::cerr << "Неизвестный preset: " << DEFAULT_PRESET << std::endl;
        return 1;
    }

    ModelPreset runtime = preset_it->second;

    const std::filesystem::path images_path = DEFAULT_IMAGES_DIR.empty()
        ? (std::filesystem::current_path() / "photo")
        : std::filesystem::path(DEFAULT_IMAGES_DIR);
    const std::filesystem::path output_path = DEFAULT_OUTPUT_DIR.empty()
        ? (std::filesystem::current_path() / "tags")
        : std::filesystem::path(DEFAULT_OUTPUT_DIR);

    if (!std::filesystem::exists(runtime.weights))
    {
        std::cerr << "Ошибка открытия весов: " << runtime.weights << std::endl;
        return 1;
    }

    if (!std::filesystem::exists(runtime.labels))
    {
        std::cerr << "Ошибка открытия файла меток: " << runtime.labels << std::endl;
        return 1;
    }

    if (!std::filesystem::exists(images_path))
    {
        std::cerr << "Ошибка доступа к папке с изображениями: " << images_path << std::endl;
        return 1;
    }

    std::error_code out_ec;
    std::filesystem::create_directories(output_path, out_ec);
    if (!std::filesystem::exists(output_path))
    {
        std::cerr << "Ошибка доступа к папке для сохранения: " << output_path << std::endl;
        return 1;
    }

    std::vector<std::string> class_list;
    if (!loadClassList(runtime.labels, class_list))
    {
        std::cerr << "Не удалось загрузить список классов." << std::endl;
        return 1;
    }

    cv::dnn::Net net = cv::dnn::readNet(runtime.weights);
    if (net.empty())
    {
        std::cerr << "Не удалось загрузить модель: " << runtime.weights << std::endl;
        return 1;
    }

    const std::vector<std::string> output_layers = net.getUnconnectedOutLayersNames();
    const InferenceParams params{ runtime.input_size, runtime.conf_threshold, runtime.nms_threshold };

    size_t processed_images = 0;
    for (const auto& entry : std::filesystem::directory_iterator(images_path))
    {
        if (!entry.is_regular_file() || !isSupportedImage(entry.path()))
        {
            continue;
        }

        cv::Mat frame = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (frame.empty())
        {
            std::wcerr << L"Ошибка чтения изображения: " << entry.path().wstring() << std::endl;
            continue;
        }

        std::vector<cv::Mat> detections = pre_process(frame, net, params, output_layers);
        Detection detection_photo = post_process(frame, detections, class_list, params);
        cv::Mat img = drawDetections(frame, detection_photo.boxes, detection_photo.confidences, detection_photo.class_ids, class_list);

        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000.0;
        double t = net.getPerfProfile(layersTimes) / freq;
        char time_label[64];
        snprintf(time_label, sizeof(time_label), "Inference time: %.2f ms", t);
        cv::putText(img, time_label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);

        const std::string main_tag = detection_photo.class_names.empty() ? "no_detection" : detection_photo.class_names.front();
        saveProcessedImage(img, main_tag, output_path);
        ++processed_images;
    }

    std::cout << "Обработано изображений: " << processed_images << std::endl;
    return 0;
}