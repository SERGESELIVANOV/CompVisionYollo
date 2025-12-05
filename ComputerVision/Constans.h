#pragma once
// Константы описывающие двоичный обьект
// Размер
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
// Фильтрация оценнок классов с малой вероятностью 
const float SCORE_THRESHOLD = 0.5;
// Для перекрявающих рамки, что бы их убрать
const float NMS_THRESHOLD = 0.45;
// Фильтрация мало вероятные обнаружения 
const float CONFIDENCE_THRESHOLD = 0.45;

// Параметры текста тегов
const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;
// Цвета для тегов
cv::Scalar BLACK = cv::Scalar(0, 0, 0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0, 0, 255);
