imageFolderPath = '/Users/putragandadewata/Desktop/PCD_20Jun/acne4/train';
csvFilePath = '/Users/putragandadewata/Desktop/PCD_20Jun/acne4/train/_classes.csv';
testRatio = 0.2; % 20% of data will be used for testing
[accuracy, confusionMatrices, binaryModels] = train_and_test_multilabel_knn(imageFolderPath, csvFilePath, testRatio);