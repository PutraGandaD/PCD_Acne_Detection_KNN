function [accuracy, confusionMatrices, binaryModels] = train_and_test_multilabel_knn(imageFolderPath, csvFilePath, testRatio)
    % Read the CSV file with preserved variable names
    opts = detectImportOptions(csvFilePath, 'VariableNamingRule', 'preserve');
    data = readtable(csvFilePath, opts);
    
    % Extract class names from CSV file
    classNames = data.Properties.VariableNames(2:end); % Assuming first column is filenames
    
    % Initialize arrays for features and labels
    numImages = height(data);
    features = [];
    labels = [];
    validImageCount = 0;
    
    % Process each image and extract features
    for i = 1:numImages
        % Get image filename
        imgFile = fullfile(imageFolderPath, strtrim(data.filename{i}));
        
        if isfile(imgFile)
            % Read image
            img = imread(imgFile);
            
            % Extract HSV features
            hsvHist = extractHSVFeatures(img);
            
            % Extract GLCM features
            glcmFeatures = extractGLCMFeatures(img);
            
            % Combine features
            imgFeatures = [hsvHist, glcmFeatures];
            
            % Append features
            features = [features; imgFeatures];
            
            % Append labels
            labels = [labels; data{i, 2:end}];
            
            % Increment valid image count
            validImageCount = validImageCount + 1;
        else
            warning('Image file %s not found. Skipping...', imgFile);
        end
    end
    
    % Convert labels to table
    labels = array2table(labels, 'VariableNames', classNames);
    
    % Initialize cell array to hold binary models and confusion matrices
    binaryModels = cell(1, length(classNames));
    confusionMatrices = cell(1, length(classNames));
    accuracies = zeros(1, length(classNames));
    
    % Train one-vs-all binary classifiers
    for i = 1:length(classNames)
        binaryLabels = categorical(labels{:, i});
        cv = cvpartition(binaryLabels, 'HoldOut', testRatio);
        trainingIdx = training(cv);
        testIdx = test(cv);
        
        trainingFeatures = features(trainingIdx, :);
        trainingLabels = binaryLabels(trainingIdx);
        testFeatures = features(testIdx, :);
        testLabels = binaryLabels(testIdx);
        
        % Train KNN model for current class
        Mdl = fitcknn(trainingFeatures, trainingLabels, 'NumNeighbors', 5);
        binaryModels{i} = Mdl;
        
        % Predict and evaluate
        predictedLabels = predict(Mdl, testFeatures);
        accuracies(i) = sum(predictedLabels == testLabels) / length(testLabels);
        confusionMatrices{i} = confusionmat(testLabels, predictedLabels);
    end
    
    % Calculate average accuracy
    accuracy = mean(accuracies);
    
    % Save the trained models
    save('knnMultiLabelModels.mat', 'binaryModels', 'classNames');
    
    fprintf('Training complete. Valid images processed: %d\n', validImageCount);
    fprintf('Average model accuracy: %.2f%%\n', accuracy * 100);
end

function hsvHist = extractHSVFeatures(img)
    % Convert image to HSV
    hsvImg = rgb2hsv(img);
    
    % Compute histogram for each channel
    hHist = imhist(hsvImg(:,:,1), 16);
    sHist = imhist(hsvImg(:,:,2), 16);
    vHist = imhist(hsvImg(:,:,3), 16);
    
    % Normalize histograms
    hHist = hHist / sum(hHist);
    sHist = sHist / sum(sHist);
    vHist = vHist / sum(vHist);
    
    % Concatenate histograms to form the feature vector
    hsvHist = [hHist; sHist; vHist]';
end

function glcmFeatures = extractGLCMFeatures(img)
    % Convert image to grayscale
    grayImg = rgb2gray(img);
    
    % Compute GLCM
    glcm = graycomatrix(grayImg, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
    
    % Compute statistics from GLCM
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    % Form the feature vector
    glcmFeatures = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];
end
