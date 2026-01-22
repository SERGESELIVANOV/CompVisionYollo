#include "GUI.h"
#include <QMessageBox>
#include <QStandardPaths>

GUI::GUI(QWidget* parent)
    : QMainWindow(parent), m_detector(nullptr), m_isProcessing(false)
{
    ui.setupUi(this);

    // Подключаем сигналы к слотам
    connect(ui.inputDirButton, &QPushButton::clicked, this, &GUI::on_inputDirButton_clicked);
    connect(ui.outputDirButton, &QPushButton::clicked, this, &GUI::on_outputDirButton_clicked);
    connect(ui.startButton, &QPushButton::clicked, this, &GUI::on_startButton_clicked);

    // Устанавливаем начальный текст в полях
    ui.inputDirEdit->setText("C:/Users/polezhaev/Desktop/Materials/photo");
    ui.outputDirEdit->setText(QDir::currentPath() + "/ComputerVision/tags");

    logMessage("Приложение запущено. Выберите папки и нажмите 'Запустить обработку'");
}

GUI::~GUI()
{
    if (m_detector) {
        delete m_detector;
    }
}

void GUI::on_inputDirButton_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, "Выберите папку с изображениями",
        ui.inputDirEdit->text());
    if (!dir.isEmpty()) {
        ui.inputDirEdit->setText(dir);
    }
}

void GUI::on_outputDirButton_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, "Выберите папку для результатов",
        ui.outputDirEdit->text());
    if (!dir.isEmpty()) {
        ui.outputDirEdit->setText(dir);
    }
}

void GUI::on_startButton_clicked()
{
    if (m_isProcessing) {
        return; // Уже обрабатываем
    }

    QString inputDir = ui.inputDirEdit->text();
    QString outputDir = ui.outputDirEdit->text();

    if (inputDir.isEmpty()) {
        QMessageBox::warning(this, "Ошибка", "Выберите папку с изображениями!");
        return;
    }

    if (outputDir.isEmpty()) {
        QMessageBox::warning(this, "Ошибка", "Выберите папку для результатов!");
        return;
    }

    QDir inputDirCheck(inputDir);
    if (!inputDirCheck.exists()) {
        QMessageBox::warning(this, "Ошибка", "Папка с изображениями не существует!");
        return;
    }

    // Создаем детектор объектов
    if (m_detector) {
        delete m_detector;
    }
    m_detector = new ObjectDetector(this);

    connect(m_detector, &ObjectDetector::progressUpdated, this, &GUI::onProgressUpdated);
    connect(m_detector, &ObjectDetector::logMessage, this, &GUI::onLogMessage);
    connect(m_detector, &ObjectDetector::processingFinished, this, &GUI::onProcessingFinished);

    QString modelType = getModelPreset();

    setControlsEnabled(false);
    ui.progressBar->setValue(0);
    ui.logTextEdit->clear();
    m_isProcessing = true;

    logMessage("Запуск обработки изображений...");
    logMessage("Входная папка: " + inputDir);
    logMessage("Выходная папка: " + outputDir);
    logMessage("Модель: " + ui.modelComboBox->currentText());

    // Инициализируем детектор
    if (!m_detector->initialize(modelType, inputDir, outputDir)) {
        setControlsEnabled(true);
        m_isProcessing = false;
        QMessageBox::critical(this, "Ошибка", "Не удалось инициализировать детектор объектов!");
        return;
    }

    // Запускаем обработку
    m_detector->processImages();
}

void GUI::onProgressUpdated(int percentage)
{
    ui.progressBar->setValue(percentage);
}

void GUI::onLogMessage(const QString& message)
{
    logMessage(message);
}

void GUI::onProcessingFinished(bool success, const QString& message)
{
    setControlsEnabled(true);
    m_isProcessing = false;
    ui.progressBar->setValue(success ? 100 : 0);

    if (success) {
        logMessage("Обработка завершена успешно!");
        QMessageBox::information(this, "Готово", message);
    }
    else {
        logMessage("Обработка завершилась с ошибкой: " + message);
        QMessageBox::warning(this, "Ошибка", "Во время обработки произошла ошибка!\n"
            "Проверьте логи для получения подробной информации.");
    }
}

void GUI::logMessage(const QString& message)
{
    ui.logTextEdit->append(message);
    ui.logTextEdit->ensureCursorVisible();
}

void GUI::setControlsEnabled(bool enabled)
{
    ui.inputDirButton->setEnabled(enabled);
    ui.outputDirButton->setEnabled(enabled);
    ui.startButton->setEnabled(enabled);
    ui.modelComboBox->setEnabled(enabled);
}

QString GUI::getModelPreset() const
{
    // Возвращает preset на основе выбранной модели
    if (ui.modelComboBox->currentIndex() == 0) {
        return "yolo11";
    }
    else {
        return "yolov5";
    }
}