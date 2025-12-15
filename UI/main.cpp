#include "mainwindow.h"
#include <QApplication>
#include <QStyleFactory>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // Настройка стиля приложения
    app.setStyle(QStyleFactory::create("Fusion"));

    // Устанавливаем информацию о приложении
    app.setApplicationName("Photo Analyzer");
    app.setOrganizationName("YourCompany");
    app.setApplicationVersion("1.0");

    MainWindow window;
    window.show();

    return app.exec();
}
