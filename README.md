# Orange-Detection

## AIM AND OBJECTIVES
## AIM
#### To create a classification system which detects the quality of a orange and specifies whether the given orange is rotten or not.


## ABSTRACT

This project focused on the development of a deep learning model based on the YOLOv5 architecture to classify fresh and rotten oranges. By collecting and preprocessing a dataset of orange images, training the YOLOv5 model, and evaluating its performance, the project aimed to provide an efficient solution for quality control in the fruit industry. The results
highlighted the model's ability to accurately distinguish between fresh and rotten oranges, showcasing its potential for practical applications. Fruit classification in deep learning has
several significant applications and is essential for various industries. Fruit classification in involves training models to automatically categorize and distinguish different types of fruits based on their visual features. Fruit classification is an interesting application of computer vision. Traditional fruit classification methods have often relied on manual operations based on visual ability and such methods are tedious, time consuming and inconsistent. External shape appearance is the main source for fruit classification. In recent years, computer machine vision and image processing techniques have been found increasingly useful in the fruit industry, especially for applications in quality inspection and color, size, shape sorting. Researches in this area indicate the feasibility of using machine vision systems to improve product quality while freeing people from the traditional hand sorting of fruits. This paper deals various image processing techniques used for fruit classification. Keywords- Fruit, Feature Extraction, Neural Network, Convolution Neural Network (CNN),Fruit Classification Yolov5; orange fruit classiﬁcation


## INTRODUCTION
Quality control is essential in the fruit industry to ensure the delivery of fresh and safe products to consumers. With the advancement of deep learning techniques, the application of
the YOLOv5 model for fruit quality assessment has gained prominence. This report details the process of leveraging the YOLOv5 architecture for fresh and rotten orange classification,
emphasizing its significance in maintaining product quality and safety standards. Image classification and object detection are two related but distinct tasks in the field of computer
vision.

### 1. Image Classification:•

Image classification involves categorizing an entire image into a specific class or category. The task is to assign a label to the entire image, indicating what is contained within it.
The model learns to recognize patterns and features within the entire image to make predictions. Commonly used models for image classification include deep learning architectures such as convolutional neural networks (CNNs). Applications of image classification include identifying objects in images, recognizing handwritten digits, or classifying images into various categories such as animals, vehicles, or natural scenes.

### 2. Object Detection:

Object detection is the task of not only categorizing objects within an image but also locating and delineating their boundaries with bounding boxes. Object detection models can identify and localize multiple objects within a single image and classify them into different categories simultaneously. Popular object detection models include region-based CNNs (R-CNN), You Only Look Once (YOLO), and Single Shot Multibox Detector (SSD). Object detection finds applications in various fields, including autonomous driving,
surveillance, and industrial quality control, where the precise localization and identification of objects are crucial. While both image classification and object detection are essential tasks in computer vision, object detection goes beyond simple classification by providing more detailed information about the location of objects within an image. Both tasks are essential for a wide range of applications, from everyday image recognition to advanced robotics and AI systems.


## LITERATURE REVIEW

Prior research has demonstrated the utility of deep learning models in various quality control
applications, including fruit classification and defect detection. YOLOv5, known for its real-
time object detection capabilities and high accuracy, has been effectively applied in similar
tasks, making it a suitable candidate for the classification of fresh and rotten oranges.

## PROPOSED SYSTEM

1] Study basics of machine learning and image recognition.

2] Start with implementation

        A. Front-end development
        B. Back-end development
        
3] Testing, analyzing and improvising the model. An application using python IDLE and its machine learning libraries will be using machine learning to identify whether a given orange is rotten or not.

4] use datasets to interpret the object and suggest whether a given orange on the camera’s viewfinder is rotten or not.


## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers everywhere.
    
• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.
    
• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.


## Jetson Nano 2GB







![Jetson-nano-labeled-01](https://github.com/nehachandekar/Orange-Detection/assets/149763129/2768b807-5ee5-46ec-8e16-5bb79d86faf3)





## Installation

#### Initial Configuration

```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*

```
#### Create Swap 
```bash
udo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
# make entry in fstab file
/swapfile1	swap	swap	defaults	0 0
```
#### Cuda env in bashrc
```bash
vim ~/.bashrc

# add this lines
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

```
#### Update & Upgrade
```bash
sudo apt-get update
sudo apt-get upgrade
```
#### Install some required Packages
```bash
sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo apt-get install libopenblas-base libopenmpi-dev
```
#### Install Torch
```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" 
sudo python3 -c "import torch; print(torch.cuda.is_available())"
```
#### Install Torchvision
```bash
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
```
#### Clone Yolov5 
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
```
#### Download weights and Test Yolov5 Installation on USB webcam
```bash
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt  --source 0
```
## Orange Dataset Training

### We used Google Colab And Roboflow

#### train your model on colab and download the weights and pass them into yolov5 folder.


## Running orange Detection Model
source '0' for webcam

```bash
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0
```
## Demo



https://github.com/nehachandekar/Orange-Detection/assets/149763129/ed793edc-e91b-4a0a-b623-0960153cf475











## METHODOLOGY

### • Yolov5 Network Model

The YOLOv5 architecture was selected for its efficiency in object detection and classification tasks. With a streamlined design and improved performance, YOLOv5 facilitated the accurate
identification and classification of fresh and rotten oranges in the dataset. Classifying oranges using YOLOv5 involves training a model to detect and classify images containing oranges.YOLOv5's object detection capabilities make it suitable for this task. Here is an introduction to orange classification using YOLOv5:

1. Data Collection and Annotation: Gather a diverse dataset of orange images, including fresh and rotten oranges in various conditions, lighting, and backgrounds. Annotate the images with bounding boxes and corresponding class labels specifically for oranges.

2. Installation of YOLOv5: Install YOLOv5 using the provided installation instructions, ensuring that the necessary dependencies are met. You can install it using pip:

3. Dataset Preparation: Organize the dataset into training and validation sets. Ensure that the dataset is well-prepared and structured according to YOLOv5's requirements.

4. Model Training: Train the YOLOv5 model using the annotated dataset. Use the provided training script, specifying the dataset directory, batch size, and the number of epochs for training. Monitor the training process to ensure that the model is effectively learning to detect and classify oranges.

5. Evaluation and Validation: Evaluate the trained model using the validation set. Calculate metrics such as precision, recall, and mean Average Precision (mAP) to assess the model's performance in accurately identifying and classifying oranges.

6. Inference and Deployment: Use the trained YOLOv5 model to perform inference on new images containing oranges. Adjust the confidence threshold to control the precision-recall trade-off in the predictions. Deploy the model for real-time orange classification tasks or integrate it into an existing application.

7. Fine-Tuning and Optimization: Fine-tune the YOLOv5 model and optimize its hyperparameters to improve its accuracy and robustness. Experiment with data augmentation, transfer learning, and regularization techniques to enhance the model's performance, especially when dealing with varying lighting conditions and backgrounds. Orange classification using YOLOv5 can be useful in various domains, including agricultural production, food processing, and quality control, where accurately identifying and sorting oranges is essential for efficient operations and maintaining product quality.

## DATASET

For training and testing, all the pictures were chosen from Fruits fresh and rotten for classification dataset, which is publicly available on Kaggle. The dataset contains 3 different
fruits pictures (Mango, Banana, Orange) of 2 classes (Rotten and Fresh). Each class represents one type of fruit. Here for classification we have used Orange fruit. These classes
are chosen because some fruits have similar appearances and are frequently bought in retail markets. Each class consists of fresh oranges and rotten oranges images.The dataset include training and testing dataset, where for training it has freshoranges (1466 files) and rottenoranges (1595 files) and for testing freshoranges (388 files) and rottenoranges (403
files) To ensure diversity, images were captured under different lighting conditions,backgrounds, and angles, simulating real-world scenarios. After removing the background, all
the fruits were resized to 416×416 pixels of standard RGB pictures. The augmented dataset was split in the ratio of 70:20:10 for training, testing, and validation purposes and then training process with 200 epochs, a batch size of 64

## Dataset Properties

1. Training set size: 2696 images

2. Validation set size: 771 images

3. Test set size: 385 Images

4. Number of classes: 2

5. Image size: 416*416 Pixels

In YOLOv5, during the training process, the loss function is comprised of several components, each serving a specific purpose in guiding the model's learning process. These components include the box loss, class (cls) loss, and objectness (obj) loss:

1. Box Loss (Localization Loss): The box loss measures the discrepancy between the predicted bounding box coordinates and the ground truth bounding box coordinates. It ensures that the model accurately localizes the objects in the image by minimizing the differences between the predicted and actual bounding box parameters.

2. Class Loss: The class loss penalizes the model based on the difference between thepredicted class probabilities and the actual class labels. It ensures that the model correctly classifies the detected objects into their respective categories, contributing to the overall accuracy of the object detection task.

3. Objectness Loss: The objectness loss penalizes the model based on the presence or absence of objects within a grid cell. It helps the model to identify grid cells that contain objects, emphasizing the importance of object presence in the training process. This loss contributes to the model's ability to detect objects accurately in the image. By incorporating these three components into the loss function, YOLOv5 effectively guides the model to learn the necessary object localization and classification tasks during the training phase. Minimizing the box loss, class loss, and objectness loss collectively contributes to the model's ability to detect and classify objects with high accuracy during inference. Precision measures how much of the bbox predictions are correct ( True positives / (True positives + False positives)), and Recall measures how much of the true bbox were correctly predicted ( True positives / (True positives + False negatives)). ‘mAP_0.5’ is the mean Average Precision (mAP) at IoU (Intersection over Union) threshold of 0.5. ‘ mAP_0.5:0.99’ is the average mAP over different IoU thresholds, ranging from 0.5 to 0.99.

4. ResultsThe trained YOLOv5 model demonstrated an impressive accuracy of 99% in classifying fresh and rotten oranges. Precision, recall, and F1 score were computed to be 0.993, 0.993, and 0.995, respectively, highlighting the model's robustness in accurately differentiating between fresh and rotten oranges. The mAP score for the classification task was determined to be 0.995, underscoring the model's reliability in accurately localizing and categorizing the two types of oranges.

## Discussion
The results underscore the YOLOv5 model's efficacy in the classification of fresh and rotten oranges, emphasizing its potential for automating quality control processes in the fruit industry. Despite the model's high accuracy, challenges such as variations in lighting and occlusions may affect its performance in real-world environments. Further enhancements in
data preprocessing and model refinement could potentially address these challenges, enhancing the model's robustness and generalization capabilities.

## Conclusion
The project successfully demonstrated the feasibility of utilizing the YOLOv5 model for fresh and rotten orange classification, highlighting its potential for streamlining quality
control processes in the fruit industry. By accurately identifying and distinguishing between fresh and rotten oranges, the model contributes to maintaining product quality standards and
ensuring consumer satisfaction. Future research should focus on enhancing the model's adaptability to different environmental conditions and expanding its applicability to other fruit types and quality assessment tasks.

## References
1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR).

2. Wang, C., Zhang, H., Lian, J., & Zhang, L. (2020). YOLOv5: A Better, Faster, Stronger PyTorch. arXiv preprint arXiv:2011.08036.





