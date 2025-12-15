#pragma once
// Параметры входного изображения
// Размер
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
// Порог вероятности бокса и для NMS 
const float SCORE_THRESHOLD = 0.5;
// Для отрисовки подписи, если не задать
const float NMS_THRESHOLD = 0.45;
// Порог уверенности итоговый
const float CONFIDENCE_THRESHOLD = 0.5;
// Настройки шрифта
const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;
// Цвета для подписей
cv::Scalar BLACK = cv::Scalar(0, 0, 0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0, 0, 255);
