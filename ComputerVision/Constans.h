#pragma once
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

// Цвета для отрисовки (BGR формат в OpenCV)
cv::Scalar BLACK = cv::Scalar(0, 0, 0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0, 0, 255);
