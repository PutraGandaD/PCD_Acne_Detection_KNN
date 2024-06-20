imageFilePath = '/Users/putragandadewata/Desktop/PCD_20Jun/acne1/tes_klasifikasi/purulant.jpg';
modelFilePath = '/Users/putragandadewata/Desktop/PCD_20Jun/acne4/knnMultiLabelModels.mat';
csvFilePath = '/Users/putragandadewata/Desktop/PCD_20Jun/acne4/train/_classes.csv';
predict_multilabel(imageFilePath, modelFilePath, csvFilePath);