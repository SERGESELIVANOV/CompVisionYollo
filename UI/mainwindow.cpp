#include "mainwindow.h"
#include "ui_mainwindow.h"  // Этот файл создается автоматически Qt Designer
#include "photoprocessor.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QImageReader>
#include <QPixmap>
#include <QThreadPool>
#include <QScrollArea>
#include <QSplitter>
#include <QTextEdit>
#include <QStatusBar>
#include <QApplication>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_processedCount(0)
    , m_btnSelectFolder(nullptr)
    , m_btnStartAnalysis(nullptr)
    , m_progressBar(nullptr)
    , m_statusLabel(nullptr)
    , m_listWidget(nullptr)
    , m_imageLabel(nullptr)
    , m_resultText(nullptr)
    , m_countLabel(nullptr)
{
    ui->setupUi(this);
    setupUI();
    setupConnections();

    setWindowTitle("Photo Analyzer v1.0");
    resize(1200, 700);
}

void MainWindow::setupUI()
{
    // Создаем центральный виджет
    QWidget *centralWidget = new QWidget(this);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    // 1. Панель управления
    QHBoxLayout *controlLayout = new QHBoxLayout();

    m_btnSelectFolder = new QPushButton("Выбрать папку", this);
    m_btnSelectFolder->setFixedSize(150, 40);

    m_btnStartAnalysis = new QPushButton("Начать анализ", this);
    m_btnStartAnalysis->setFixedSize(150, 40);
    m_btnStartAnalysis->setEnabled(false);

    m_progressBar = new QProgressBar(this);
    m_progressBar->setFixedHeight(30);
    m_progressBar->setTextVisible(true);

    m_statusLabel = new QLabel("Готово к работе", this);
    m_statusLabel->setFixedWidth(200);

    controlLayout->addWidget(m_btnSelectFolder);
    controlLayout->addWidget(m_btnStartAnalysis);
    controlLayout->addWidget(m_progressBar);
    controlLayout->addWidget(m_statusLabel);
    controlLayout->addStretch();

    // 2. Основная область
    QSplitter *splitter = new QSplitter(Qt::Horizontal, this);

    // Левая часть
    QWidget *leftWidget = new QWidget(splitter);
    QVBoxLayout *leftLayout = new QVBoxLayout(leftWidget);

    QLabel *filesLabel = new QLabel("Фотографии:", this);
    m_listWidget = new QListWidget(this);
    m_listWidget->setSelectionMode(QAbstractItemView::SingleSelection);

    leftLayout->addWidget(filesLabel);
    leftLayout->addWidget(m_listWidget);

    // Правая часть
    QWidget *rightWidget = new QWidget(splitter);
    QVBoxLayout *rightLayout = new QVBoxLayout(rightWidget);

    QLabel *previewLabel = new QLabel("Предпросмотр", this);
    previewLabel->setAlignment(Qt::AlignCenter);

    m_imageLabel = new QLabel(this);
    m_imageLabel->setAlignment(Qt::AlignCenter);
    m_imageLabel->setMinimumSize(400, 400);
    m_imageLabel->setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;");

    QLabel *resultLabel = new QLabel("Результаты анализа:", this);
    m_resultText = new QTextEdit(this);
    m_resultText->setReadOnly(true);
    m_resultText->setMaximumHeight(150);

    rightLayout->addWidget(previewLabel);
    rightLayout->addWidget(m_imageLabel, 1);
    rightLayout->addWidget(resultLabel);
    rightLayout->addWidget(m_resultText);

    splitter->addWidget(leftWidget);
    splitter->addWidget(rightWidget);
    splitter->setSizes(QList<int>() << 400 << 800);

    // 3. Статусбар
    QStatusBar *statusBar = new QStatusBar(this);
    m_countLabel = new QLabel("Файлов: 0", this);
    statusBar->addWidget(m_countLabel);

    // Собираем интерфейс
    mainLayout->addLayout(controlLayout);
    mainLayout->addWidget(splitter);
    mainLayout->addWidget(statusBar);

    setCentralWidget(centralWidget);
}

void MainWindow::setupConnections()
{
    connect(m_btnSelectFolder, &QPushButton::clicked,
            this, &MainWindow::onSelectFolder);
    connect(m_btnStartAnalysis, &QPushButton::clicked,
            this, &MainWindow::onStartAnalysis);
    connect(m_listWidget, &QListWidget::itemClicked,
            this, &MainWindow::onPhotoSelected);
}

void MainWindow::onSelectFolder()
{
    QString folderPath = QFileDialog::getExistingDirectory(
        this,
        "Выберите папку с фотографиями",
        QDir::homePath(),
        QFileDialog::ShowDirsOnly
        );

    if (folderPath.isEmpty()) return;

    m_listWidget->clear();
    m_photoPaths.clear();

    // Поддерживаемые форматы изображений
    QStringList filters;
    filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp" << "*.gif"
            << "*.tiff" << "*.webp";

    QDir directory(folderPath);
    QStringList files = directory.entryList(filters, QDir::Files, QDir::Name);

    foreach (const QString &file, files) {
        QString fullPath = directory.absoluteFilePath(file);
        m_photoPaths.append(fullPath);

        QListWidgetItem *item = new QListWidgetItem(file);
        item->setData(Qt::UserRole, fullPath);
        m_listWidget->addItem(item);
    }

    m_countLabel->setText(QString("Файлов: %1").arg(m_photoPaths.count()));
    m_btnStartAnalysis->setEnabled(m_photoPaths.count() > 0);
}

void MainWindow::onPhotoSelected(QListWidgetItem *item)
{
    if (!item) return;

    QString filePath = item->data(Qt::UserRole).toString();

    QPixmap pixmap(filePath);
    if (!pixmap.isNull()) {
        // Масштабируем для предпросмотра
        QPixmap scaled = pixmap.scaled(
            m_imageLabel->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
            );
        m_imageLabel->setPixmap(scaled);

        // Показываем информацию о файле
        QFileInfo info(filePath);
        QString fileInfo = QString(
                               "Файл: %1\n"
                               "Размер: %2 KB\n"
                               "Разрешение: %3x%4\n"
                               "Последнее изменение: %5"
                               ).arg(info.fileName())
                               .arg(info.size() / 1024)
                               .arg(pixmap.width())
                               .arg(pixmap.height())
                               .arg(info.lastModified().toString("dd.MM.yyyy hh:mm"));

        m_resultText->setText(fileInfo);
    } else {
        m_imageLabel->setText("Не удалось загрузить изображение");
        m_resultText->setText("Ошибка загрузки файла");
    }
}

void MainWindow::onStartAnalysis()
{
    if (m_photoPaths.isEmpty()) return;

    // Блокируем кнопки на время анализа
    m_btnStartAnalysis->setEnabled(false);
    m_btnSelectFolder->setEnabled(false);
    m_statusLabel->setText("Анализ...");
    m_progressBar->setRange(0, m_photoPaths.count());
    m_progressBar->setValue(0);
    m_processedCount = 0;

    // Создаем и запускаем обработчик
    PhotoProcessor *processor = new PhotoProcessor(m_photoPaths);

    connect(processor, &PhotoProcessor::photoProcessed,
            this, &MainWindow::onPhotoProcessed);
    connect(processor, &PhotoProcessor::finished,
            this, &MainWindow::onAnalysisFinished);
    connect(processor, &PhotoProcessor::finished,
            processor, &QObject::deleteLater);

    // Запускаем в отдельном потоке
    QThread *thread = new QThread;
    processor->moveToThread(thread);

    connect(thread, &QThread::started, processor, &PhotoProcessor::processPhotos);
    connect(processor, &PhotoProcessor::finished, thread, &QThread::quit);
    connect(thread, &QThread::finished, thread, &QThread::deleteLater);

    thread->start();
}

void MainWindow::onPhotoProcessed(int index, const QString &result)
{
    m_processedCount++;
    m_progressBar->setValue(m_processedCount);

    // Обновляем элемент в списке
    if (index < m_listWidget->count()) {
        QListWidgetItem *item = m_listWidget->item(index);
        QString text = item->text();

        // Убираем предыдущую отметку если есть
        if (text.endsWith(" ✓")) {
            text = text.left(text.length() - 2);
        }

        item->setText(text + " ✓");
        item->setData(Qt::UserRole + 1, result); // Сохраняем результат

        // Если это выбранный элемент, обновляем информацию
        if (item->isSelected()) {
            m_resultText->append("Результат анализа: " + result);
        }
    }
}

void MainWindow::onAnalysisFinished()
{
    m_btnStartAnalysis->setEnabled(true);
    m_btnSelectFolder->setEnabled(true);
    m_statusLabel->setText("Анализ завершен");

    QMessageBox::information(
        this,
        "Готово",
        QString("Анализ завершен!\nОбработано фотографий: %1").arg(m_processedCount)
        );
}

MainWindow::~MainWindow()
{
    delete ui;
}
