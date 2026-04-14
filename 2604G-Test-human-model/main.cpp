#include "MainWindow.h"

#include <QApplication>
#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[]) {
    try {
        QApplication qapp(argc, argv);

        MainWindow mainWin;
        mainWin.show();

        return qapp.exec();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
