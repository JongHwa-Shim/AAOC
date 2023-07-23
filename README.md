# Auxiliary Augmentation Operation Classifier
Abstract
Although a generative adversarial network (GAN) can generate realistic and distinct images, it requires numerous training data. Data augmentation is a popular method of incrementing data using various augmentation operations. However, data augmentation for training GANs can lead to augmentation leaks where unintended distortion effects in augmented images appear in images generated by the GAN. This augmentation leaks occur because the GAN is trained to generate images similar to the training dataset containing augmented images. Although recent studies have revealed that maintaining the invertibility of data augmentation can prevent augmentation leaks, maintaining it for various data augmentation configurations and datasets is challenging. This paper proposes a novel auxiliary neural network called an auxiliary augmentation operation classifier (AAOC) to prevent augmentation leaks. This network learns the classification of augmentation operations during GAN training, guiding the GAN to avoid generating images with unintended distortions. Due to its simple and lightweight architecture, the AAOC does not increase the training time of GANs. Extensive experiments on the three public datasets revealed that the AAOC reduced the occurrence of augmentation leaks by a factor of 30 compared to the basic method and improved image quality by 50% for leak-free images.
Keywords: Data augmentation, generative adversarial network (GAN), deep learning, augmentation leak, auxiliary classifier