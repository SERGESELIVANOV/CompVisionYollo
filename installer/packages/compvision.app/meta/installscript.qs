function Component()
{
    // Конструктор компонента
}

Component.prototype.createOperations = function()
{
    component.createOperations();
    
    // Создаем ярлык на рабочем столе
    if (systemInfo.productType === "windows") {
        component.addOperation("CreateShortcut", 
            "@TargetDir@/GUI.exe", 
            "@DesktopDir@/Computer Vision YOLO.lnk",
            "workingDirectory=@TargetDir@",
            "iconPath=@TargetDir@/GUI.exe",
            "iconId=0",
            "description=Computer Vision Object Detection");
        
        // Добавляем в меню Пуск с иконкой
        component.addOperation("CreateShortcut", 
            "@TargetDir@/GUI.exe", 
            "@StartMenuDir@/Computer Vision YOLO.lnk",
            "workingDirectory=@TargetDir@",
            "iconPath=@TargetDir@/GUI.exe", 
            "iconId=0",
            "description=Computer Vision Object Detection");
            
        // Создаем ярлык для деинсталлятора
        component.addOperation("CreateShortcut", 
            "@TargetDir@/MaintenanceTool.exe", 
            "@DesktopDir@/Uninstall Computer Vision YOLO.lnk",
            "workingDirectory=@TargetDir@",
            "description=Uninstall Computer Vision YOLO");
    }
}

Component.prototype.installed = function()
{
    console.log("Computer Vision YOLO installed successfully!");
}

Component.prototype.uninstalled = function()
{
    console.log("Computer Vision YOLO uninstalled successfully!");
}