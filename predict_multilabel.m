function predict_multilabel(imageFilePath, modelFilePath, csvFilePath)
    % Load the trained models
    load(modelFilePath, 'binaryModels', 'classNames');
    
    % Read and process the new image
    img = imread(imageFilePath);
    
    % Extract HSV features
    hsvHist = extractHSVFeatures(img);
    
    % Extract GLCM features
    glcmFeatures = extractGLCMFeatures(img);
    
    % Combine features
    imgFeatures = [hsvHist, glcmFeatures];
    
    % Predict probabilities for each class
    probabilities = zeros(1, length(classNames));
    for i = 1:length(classNames)
        [label, score] = predict(binaryModels{i}, imgFeatures);
        probabilities(i) = score(2); % Probability of being class 1
    end
    
    % Output the classes with highest probabilities
    [sortedProbs, sortedIdx] = sort(probabilities, 'descend');
    fprintf('Predicted labels with probabilities:\n');
    for i = 1:length(classNames)
        fprintf('%s: %.2f%%\n', classNames{sortedIdx(i)}, sortedProbs(i) * 100);
    end
end

% Supporting functions
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
