function acnePredictionGUI()
    % Create a figure for the GUI
    hFig = figure('Name', 'Acne Prediction GUI', 'Position', [100, 100, 800, 600]);
    
    % Button to input image
    uicontrol('Style', 'pushbutton', 'String', 'Input Image', ...
              'Position', [50, 550, 100, 30], 'Callback', @inputImageCallback);
          
    % Axes to display the input image
    hAxesImage = axes('Parent', hFig, 'Position', [0.05, 0.3, 0.4, 0.4]);
    
    % Axes to display the HSV histogram
    hAxesHSV = axes('Parent', hFig, 'Position', [0.55, 0.55, 0.4, 0.4]);
    
    % Axes to display the GLCM graph
    hAxesGLCM = axes('Parent', hFig, 'Position', [0.55, 0.05, 0.4, 0.4]);
    
    % Text box to display the probabilities
    hTextProbabilities = uicontrol('Style', 'text', 'Position', [50, 450, 300, 80], ...
                                   'HorizontalAlignment', 'left', 'FontSize', 12);
    
    function inputImageCallback(~, ~)
        % Load the trained models
        modelFilePath = '/Users/putragandadewata/Desktop/PCD_20Jun/acne4/knnMultiLabelModels.mat';
        csvFilePath = '/Users/putragandadewata/Desktop/PCD_20Jun/acne4/train/_classes.csv';
        load(modelFilePath, 'binaryModels', 'classNames');
        
        % Input image file
        [fileName, filePath] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files'}, 'Select an Image');
        if isequal(fileName, 0)
            return; % No file selected
        end
        imageFilePath = fullfile(filePath, fileName);
        
        % Read and process the new image
        img = imread(imageFilePath);
        imshow(img, 'Parent', hAxesImage);
        title(hAxesImage, 'Input Image');
        
        % Extract HSV features
        hsvHist = extractHSVFeatures(img);
        
        % Extract GLCM features
        glcmFeatures = extractGLCMFeatures(img);
        
        % Combine features
        imgFeatures = [hsvHist, glcmFeatures];
        
        % Predict probabilities for each class
        probabilities = zeros(1, length(classNames));
        for i = 1:length(classNames)
            [~, score] = predict(binaryModels{i}, imgFeatures);
            probabilities(i) = score(2); % Probability of being class 1
        end
        
        % Display the probabilities
        [sortedProbs, sortedIdx] = sort(probabilities, 'descend');
        probText = 'Predicted labels with probabilities:\n';
        for i = 1:length(classNames)
            probText = [probText, sprintf('%s: %.2f%%\n', classNames{sortedIdx(i)}, sortedProbs(i) * 100)];
        end
        set(hTextProbabilities, 'String', probText);
        
        % Display HSV histogram
        bar(hAxesHSV, hsvHist);
        title(hAxesHSV, 'HSV Histogram');
        
        % Display GLCM features
        bar(hAxesGLCM, glcmFeatures);
        title(hAxesGLCM, 'GLCM Features');
        
        % Show the image with highlighted acne areas (for simplicity, using the input image)
        % In a real scenario, image processing techniques can be applied to highlight acne areas
        imshow(img, 'Parent', hAxesImage);
        title(hAxesImage, 'Highlighted Acne Areas');
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
