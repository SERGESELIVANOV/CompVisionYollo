#pragma once
#include <opencv2/opencv.hpp>

// Параметры входного изображения для модели YOLO
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;

// Пороги фильтрации детекций (используются в пресетах моделей, здесь для справки)
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.5;

// Настройки шрифта для отрисовки подписей
const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Функции для получения цветов (inline static переменные)
inline static cv::Scalar getBlackColor() { return cv::Scalar(0, 0, 0); }
inline static cv::Scalar getBlueColor() { return cv::Scalar(255, 178, 50); }
inline static cv::Scalar getYellowColor() { return cv::Scalar(0, 255, 255); }
inline static cv::Scalar getRedColor() { return cv::Scalar(0, 0, 255); }