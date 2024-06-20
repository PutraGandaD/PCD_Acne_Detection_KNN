imageFilePath = 'C:\Users\HP\Documents\Semester 6\PCDKNNS\PCD_Acne_Detection_KNN\image_classify_test\blackhead.jpg';
modelFilePath = 'C:\Users\HP\Documents\Semester 6\PCDKNNS\PCD_Acne_Detection_KNN\knnMultiLabelModels.mat';
csvFilePath = 'C:\Users\HP\Documents\Semester 6\PCDKNNS\PCD_Acne_Detection_KNN\train\_classes.csv';
predict_multilabel(imageFilePath, modelFilePath, csvFilePath);