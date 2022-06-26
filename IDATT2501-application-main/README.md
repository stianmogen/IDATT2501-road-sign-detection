# IDATT2105-application
 
### Development information

The application is developed using both emulators and physical devices. Android 7.0 Nougat or newer is required to run the application. To run the application with desired result on physical device, you would need a newer mid to high-end spec device. The application has been tested on older and newer phones of different specification for reference: 

- Smooth performance: Samsung Galaxy S9 plus www.samsung.com/smartphones/galaxy-s9/specs
- Inadequate performance: Hauwei P20 lite www.gsmarena.com/huawei_p20_lite_(2019)-9703.php

The application has a physical size of 331MB. 

### Clone project

HTTPS:
```
https://github.com/stianmogen/IDATT2501-application.git
```
SSH:
```
git@github.com:stianmogen/IDATT2501-application.git
```

### Running the application

#### Android Studio and Emulator/Hardware Device

The group has developed and tested the application using [Android Studio](https://developer.android.com/studio).

To run the application in Androdi Studio, you need to install the [Android SDK](https://developer.android.com/studio/install)

If you want to configure an Android Emulator to test the application, please see: [Managing AVDS](https://developer.android.com/studio/run/managing-avds)

If you want to run the Android Application on a physical device, please see this [Run Apps on a Hardware Device](https://developer.android.com/studio/run/device)

To run the application, you need to provide camera and locaiton permission on your device. 

#### Pytorch Modules

The application uses PyTorch modules to load and use the machine learning models. It is therefore required to have PyTorch installed when running the application.

Install PyTorch following this [guide](https://pytorch.org/get-started/locally/)

PyTorch Mobile is used in the application. Android specific Torchvision installation guide and example can be found [here](https://pytorch.org/get-started/locally/)

### Demo

![prediction_video](https://github.com/stianmogen/IDATT2501-application/blob/main/readme-utils/predictionGif.gif)
![validation_video](https://github.com/stianmogen/IDATT2501-application/blob/main/readme-utils/validationGif.gif)

Demostration is run on Huawei P20 Lite. 



