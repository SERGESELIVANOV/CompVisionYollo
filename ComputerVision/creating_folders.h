#pragma once

inline std::string saveProcessedImage(const cv::Mat& image, const std::string& tag_name, const std::filesystem::path& output_root)
{
    const std::string actual_tag = tag_name.empty() ? "no_detection" : tag_name;
    const std::filesystem::path tags_root = output_root / actual_tag;
    std::error_code ec;
    std::filesystem::create_directories(tags_root, ec);

    int counter = 1;
    while (true)
    {
        std::ostringstream ss;
        ss << actual_tag << "_" << counter++ << ".jpg";
        const std::filesystem::path candidate = tags_root / ss.str();
        if (std::filesystem::exists(candidate))
        {
            continue;
        }

        if (!cv::imwrite(candidate.string(), image))
        {
            std::cerr << "Не удалось сохранить изображение: " << candidate << std::endl;
        }
        return candidate.string();
    }
}
