#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QListWidget>
#include <QProgressBar>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QTextEdit>
#include <QStatusBar>
#include <QFileDialog>
#include <QDir>
#include <QThread>


// Вместо forward declaration используем namespace
QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class PhotoProcessor;  // Forward declaration

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onSelectFolder();
    void onStartAnalysis();
    void onPhotoProcessed(int index, const QString &result);
    void onAnalysisFinished();
    void onPhotoSelected(QListWidgetItem *item);

private:
    Ui::MainWindow *ui;
    QStringList m_photoPaths;
    int m_processedCount;

    // Виджеты как члены класса
    QPushButton *m_btnSelectFolder;
    QPushButton *m_btnStartAnalysis;
    QProgressBar *m_progressBar;
    QLabel *m_statusLabel;
    QListWidget *m_listWidget;
    QLabel *m_imageLabel;
    QTextEdit *m_resultText;
    QLabel *m_countLabel;

    void setupUI();
    void setupConnections();
};

#endif
