#pragma once
bool createDirectory(const std::string& path)
{
    return _mkdir(path.c_str()) == 0;
}

bool directoryExists(const std::string& path)
{
    struct _stat info;
    return _stat(path.c_str(), &info) == 0 && (info.st_mode & _S_IFDIR);
}

bool fileExists(const std::string& path)
{
    struct _stat info;
    return _stat(path.c_str(), &info) == 0 && (info.st_mode & _S_IFREG);
}


std::string saveProcessedImage(cv::Mat& image, const std::string& tag_name)
{
    std::filesystem::path current_path = std::filesystem::current_path();
    std::string path_str = current_path.string();
    std::string main_tags_dir = "tags";
    if (!directoryExists(main_tags_dir))
    {
        std::cout << "Создаем основную папку tags..." << std::endl;
        if (createDirectory(main_tags_dir))
        {
            std::cout << "Папка tags создана" << std::endl;
        }

    }
    else {
        std::cout << "Папка tags уже существует" << std::endl;
    }
    std::string tag_dir = path_str + "/" + main_tags_dir + "/" + tag_name;
    // Проверяем, существует ли папка с тегом
    if (!directoryExists(tag_dir))
    {
        std::cout << "Создаем папку для тега: " << tag_name << std::endl;
        if (createDirectory(tag_dir))
        {
            std::cout << "Папка для тега " << tag_name << " создана." << std::endl;
        }
    }
    if (tag_name == "no_detection")
    {
        std::string no_detection_dir = path_str + "/" + main_tags_dir + "/no_detection";

        if (!directoryExists(no_detection_dir))
        {
            std::cout << "Создаем папку для no_detection..." << std::endl;
            if (createDirectory(no_detection_dir))
            {
                std::cout << "Папка no_detection создана." << std::endl;
            }
        }
    }
    // формируем имя файла с авто-нумерацией
    int counter = 1;
    std::string filename;
    std::string full_path;
    do {
        std::ostringstream ss;
        ss << tag_name << "_" << counter << ".jpg";
        filename = ss.str();
        full_path = tag_dir + "/" + filename;
        counter++;
    } while (std::filesystem::exists(full_path));

    // сохраняем изображение
    if (cv::imwrite(full_path, image))
    {
        std::cout << " Файл сохранен: " << full_path << std::endl;
    }

    return full_path;
}
