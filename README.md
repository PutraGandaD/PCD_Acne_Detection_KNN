# Multi Class Classification for Acne Detection with KNN Algorithm using HSV and GLCM Feature Extraction

![Screen Shot 2024-06-22 at 11 47 25 PM](https://github.com/PutraGandaD/PCD_Acne_Detection_KNN/assets/54593964/bd3568f5-0d2a-4ef4-8099-16cec3db1f55)

## Anggota Grup
1. Karina (2226250058)
2. Putra Ganda Dewata (2125250069)
3. Vincent (2125250004)

## Dataset source <br>
https://universe.roboflow.com/university-of-sri-jayewardenepura-srk7m/acne-dataset-w9m3b

## Penjelasan Folder dan File
- **train** = Dataset Training, yang didalamnya terdapat file _classes.csv sebagai label dari masing - masing gambar dataset
- **train-model** = Folder yang berisikan kode (.m) untuk training model KNN
- **image_clasify_test** = Folder yang berisikan gambar untuk menguji model KNN yang sudah ditraining
- **knnMultiLabelModels.mat** = File Model KNN yang sudah ditraining dengan dataset
- **gui.m** = Run file ini untuk menjalankan program

## Cara Menjalankan Program
- **IMPORTANT STEPS :** <br>
  Sebelum run gui.m, buka gui.m dengan MATLAB atau text editor apapun dan sesuaikan terlebih dahulu path modelFilePath dan csvFilePath dengan path lokal di komputer anda.<br>
  **Untuk modelFilePath** = Copy path file knnMultiLabelModels.mat yang sudah diekstrak dari repo ini
  **Untuk csvFilePath** = Copy path file _classes.csv dari folder **train** yang sudah diekstrak dari repo ini

1. Jalankan gui.m <br>
2. Klik "Input Image" untuk load image untuk diklasifikasi oleh model. (Anda bisa menggunakan gambar dari folder **image_classify_test** untuk menguji klasifikasi model KNN.
  
