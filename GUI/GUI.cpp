#include "GUI.h"
#include <QMessageBox>
#include <QStandardPaths>
#include <QDir>

GUI::GUI(QWidget *parent)
    : QMainWindow(parent), process(nullptr)
{
    ui.setupUi(this);

    // Подключаем сигналы к слотам
    connect(ui.inputDirButton, &QPushButton::clicked, this, &GUI::on_inputDirButton_clicked);
    connect(ui.outputDirButton, &QPushButton::clicked, this, &GUI::on_outputDirButton_clicked);
    connect(ui.startButton, &QPushButton::clicked, this, &GUI::on_startButton_clicked);

    // Устанавливаем начальный текст в полях
    ui.inputDirEdit->setText("C:/Users/polezhaev/Desktop/Materials/photo");
    ui.outputDirEdit->setText("C:/Users/polezhaev/source/repos/CompVisionYollo/ComputerVision/tags");

    logMessage("Приложение запущено. Выберите папки и нажмите 'Запустить обработку'");
}

GUI::~GUI()
{
    if (process) {
        process->kill();
        process->waitForFinished(3000);
        delete process;
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

    // Проверяем существование ComputerVision.exe
    QString computerVisionPath = "C:/Users/polezhaev/source/repos/CompVisionYollo/x64/Release/ComputerVision.exe";
    if (!QFile::exists(computerVisionPath)) {
        computerVisionPath = "C:/Users/polezhaev/source/repos/CompVisionYollo/x64/Debug/ComputerVision.exe";
        if (!QFile::exists(computerVisionPath)) {
            QMessageBox::critical(this, "Ошибка", "Не найден исполняемый файл ComputerVision.exe!\n"
                                "Сначала соберите проект ComputerVision.");
            return;
        }
    }

    // Создаем процесс для запуска консольного приложения
    if (process) {
        delete process;
    }
    process = new QProcess(this);

    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &GUI::onProcessFinished);
    connect(process, &QProcess::errorOccurred, this, &GUI::onProcessError);
    connect(process, &QProcess::readyReadStandardOutput, this, &GUI::onReadyReadStandardOutput);
    connect(process, &QProcess::readyReadStandardError, this, &GUI::onReadyReadStandardError);

    // Устанавливаем переменные окружения или параметры командной строки
    QStringList arguments;
    // Здесь мы можем передать параметры через переменные окружения или аргументы командной строки
    // Пока что оставим как есть, предполагая что приложение использует жестко заданные пути

    setControlsEnabled(false);
    ui.progressBar->setValue(0);
    ui.logTextEdit->clear();

    logMessage("Запуск обработки изображений...");
    logMessage("Входная папка: " + inputDir);
    logMessage("Выходная папка: " + outputDir);
    logMessage("Модель: " + ui.modelComboBox->currentText());

    // Запускаем процесс
    process->start(computerVisionPath, arguments);
}

void GUI::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    setControlsEnabled(true);

    if (exitStatus == QProcess::NormalExit && exitCode == 0) {
        ui.progressBar->setValue(100);
        logMessage("Обработка завершена успешно!");
        QMessageBox::information(this, "Готово", "Обработка изображений завершена успешно!");
    } else {
        logMessage("Обработка завершилась с ошибкой (код: " + QString::number(exitCode) + ")");
        QMessageBox::warning(this, "Ошибка", "Во время обработки произошла ошибка!\n"
                           "Проверьте логи для получения подробной информации.");
    }
}

void GUI::onProcessError(QProcess::ProcessError error)
{
    setControlsEnabled(true);
    ui.progressBar->setValue(0);

    QString errorMessage;
    switch (error) {
        case QProcess::FailedToStart:
            errorMessage = "Не удалось запустить процесс ComputerVision.exe";
            break;
        case QProcess::Crashed:
            errorMessage = "Процесс ComputerVision.exe аварийно завершился";
            break;
        case QProcess::Timedout:
            errorMessage = "Превышено время ожидания запуска процесса";
            break;
        case QProcess::WriteError:
            errorMessage = "Ошибка записи в процесс";
            break;
        case QProcess::ReadError:
            errorMessage = "Ошибка чтения из процесса";
            break;
        default:
            errorMessage = "Неизвестная ошибка процесса";
    }

    logMessage("Ошибка: " + errorMessage);
    QMessageBox::critical(this, "Ошибка", errorMessage);
}

void GUI::onReadyReadStandardOutput()
{
    if (process) {
        QByteArray data = process->readAllStandardOutput();
        QString output = QString::fromLocal8Bit(data);
        logMessage("Вывод: " + output.trimmed());
    }
}

void GUI::onReadyReadStandardError()
{
    if (process) {
        QByteArray data = process->readAllStandardError();
        QString error = QString::fromLocal8Bit(data);
        logMessage("Ошибка: " + error.trimmed());
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
    } else {
        return "yolov5";
    }
}

