

                        Object Detection on DeepFashion dataset using DETR 
                             Advanced Computer Vision (EECE 7370)

           
Team:
Bhanu Sai Simha Vanam
Avinash Deyyam
Naga Sindhu Namana

Follow the prerequisite steps and requirements


conda create -n detr_detection python=3.7

conda activate detr_detection

pip install -r requirements.txt

-------------------------------- Training -----------------------------

1. Download DeepFashion Dataset from the official source https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok. unzip and place it in your desired path.

2. Adjust the dataset_path in the train.ipynb notebook and run all cells.
3. After training with the default training parameters for roughly 2 days, the model with be saved in detr/outputs as checkpoint.pth (I later renamed it to fashionobjectdetection.pth for readability and inference purpose)

-------------------------------- Inference -----------------------------

1. Download our trained model from https://northeastern-my.sharepoint.com/:u:/g/personal/vanam_b_northeastern_edu/EcZjvJm6vUdKinW7GVpZUdkBY9v-GppC7vJykKsq1p7PsA?e=CdhGNL and place it in your desired location
2. Adjust the model_path and test image path in the infertence.ipynb notebook and run all the cells


