#include "pch.h"
#include "creating_folders.h"
#include "pred_and_post_processes.h"


std::string saveProcessedImage(const cv::Mat& image, const std::string& tag_name, const std::filesystem::path& output_root);
std::vector<cv::Mat> pre_process(cv::Mat& input_image, cv::dnn::Net& net, const InferenceParams& params, const std::vector<std::string>& output_layers);
Detection post_process(cv::Mat& input_image, std::vector<cv::Mat>& outputs, const std::vector<std::string>& class_name, const InferenceParams& params);


class SaveProcessedImageTest : public ::testing::Test
{
protected:
    std::filesystem::path temp_dir;

    void SetUp() override
    {
        temp_dir = std::filesystem::temp_directory_path() / "save_image_test";
        std::filesystem::remove_all(temp_dir);
        std::filesystem::create_directories(temp_dir);
    }

    void TearDown() override
    {
        std::filesystem::remove_all(temp_dir);
    }
};



//Тестирование на сохранение изображения
TEST_F(SaveProcessedImageTest, SavesImageWithCorrectTag) 
{
    cv::Mat img = cv::Mat::zeros(10, 10, CV_8UC3);
    std::string result = saveProcessedImage(img, "person", temp_dir);
    std::filesystem::path result_path(result);
    // Проверка существования файла по полученому пути
    EXPECT_TRUE(std::filesystem::exists(result_path));
    //Проверка соответсвия деректорий 
    EXPECT_EQ(result_path.parent_path(), temp_dir / "person");
    // Проверка соответсвия имени файла
    EXPECT_EQ(result_path.filename(), "person_1.jpg");
}

// Проверка при попадание фото каторое не распозналось
TEST_F(SaveProcessedImageTest, EmptyTagUsesNoDetection)
{
    cv::Mat img = cv::Mat::zeros(5, 5, CV_8UC3);
    std::string result = saveProcessedImage(img, "", temp_dir);
    std::filesystem::path result_path(result);
    EXPECT_TRUE(std::filesystem::exists(result_path));
    EXPECT_EQ(result_path.parent_path(), temp_dir / "no_detection");
    EXPECT_EQ(result_path.filename(), "no_detection_1.jpg");
}

// Тест на проверку, создания правильной нумирации при сохранении файлов
TEST_F(SaveProcessedImageTest, IncrementsFileCounter)
{
    cv::Mat img = cv::Mat::zeros(5, 5, CV_8UC3);
    auto first = saveProcessedImage(img, "car", temp_dir);
    auto second = saveProcessedImage(img, "car", temp_dir);
    EXPECT_EQ(std::filesystem::path(first).filename(), "car_1.jpg");
    EXPECT_EQ(std::filesystem::path(second).filename(), "car_2.jpg");
}

TEST(OutputLayoutTest, FeaturesLastLayout)
{
    int sizes[] = { 1, 3, 85 };
    cv::Mat output(3, sizes, CV_32F);

    auto layout = analyzeOutputLayout(output, 85, 84);

    EXPECT_EQ(layout.features, 85);
    EXPECT_TRUE(layout.features_last);
    EXPECT_EQ(layout.detections, 3);
}

TEST(FlattenDetectionsTest, FeaturesLast)
{
    cv::Mat output(2, 6, CV_32F);
    float* ptr = output.ptr<float>();

    for (int i = 0; i < 12; ++i)
        ptr[i] = static_cast<float>(i);

    cv::Mat flat = flattenDetections(output, 6, true);

    ASSERT_EQ(flat.rows, 2);
    ASSERT_EQ(flat.cols, 6);
    EXPECT_FLOAT_EQ(flat.at<float>(1, 2), 8.f);
}

TEST(PostProcessTest, SingleDetection)
{
    cv::Mat image(640, 640, CV_8UC3);

    std::vector<std::string> class_names = { "elephant", "dog" };

    // 1 detection, 7 features: x y w h obj class0 class1
    cv::Mat output(1, 7, CV_32F);
    float* d = output.ptr<float>();

    d[0] = 320;  // cx
    d[1] = 320;  // cy
    d[2] = 200;  // w
    d[3] = 200;  // h
    d[4] = 1.0f; // objectness
    d[5] = 0.9f; // elephant
    d[6] = 0.1f; // dog

    std::vector<cv::Mat> outputs = { output };

    InferenceParams params;
    params.conf_threshold = 0.5f;

    Detection det = post_process(image, outputs, class_names, params);

    ASSERT_EQ(det.boxes.size(), 1);
    EXPECT_EQ(det.class_ids[0], 0);
    EXPECT_EQ(det.class_names[0], "elephant");
    EXPECT_GT(det.confidences[0], 0.8f);
}


int main(int argc, char** argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}