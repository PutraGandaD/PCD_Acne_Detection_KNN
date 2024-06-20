imageFolderPath = 'C:\Users\HP\Documents\Semester 6\PCDKNNS\PCD_Acne_Detection_KNN\train';
csvFilePath = 'C:\Users\HP\Documents\Semester 6\PCDKNNS\PCD_Acne_Detection_KNN\train\_classes.csv';
testRatio = 0.2; % 20% of data will be used for testing
[accuracy, confusionMatrices, binaryModels] = train_and_test_multilabel_knn(imageFolderPath, csvFilePath, testRatio);