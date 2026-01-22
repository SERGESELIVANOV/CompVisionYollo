function Controller()
{
    // Подавляем предупреждение о пустом имени файла
    installer.setMessageBoxAutomaticAnswer("OverwriteTargetDirectory", QMessageBox.Yes);
}

Controller.prototype.ComponentSelectionPageCallback = function()
{
    // Автоматически переходим дальше, не давая пользователю выбора
    gui.clickButton(buttons.NextButton);
};