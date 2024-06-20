function acnePredictionGUI()
    % Create a figure for the GUI
    hFig = figure('Name', 'Acne Prediction GUI', 'Position', [100, 100, 1400, 700], 'Color', [0.9 0.9 0.9]);
    
    % Panel for input image
    hPanelInput = uipanel('Title', 'Input Image', 'FontSize', 12, ...
                          'BackgroundColor', 'white', ...
                          'Position', [0.05 0.55 0.4 0.4]);
    hAxesImage = axes('Parent', hPanelInput, 'Position', [0.1, 0.1, 0.8, 0.8]);
    
    % Panel for HSV scatter plot
    hPanelHSV = uipanel('Title', 'HSV Scatter Plot', 'FontSize', 12, ...
                        'BackgroundColor', 'white', ...
                        'Position', [0.5 0.55 0.25 0.4]);
    hAxesHSV = axes('Parent', hPanelHSV, 'Position', [0.1, 0.1, 0.8, 0.8]);
    
    % Panel for HSV image with highlighted acne areas
    hPanelHSVImage = uipanel('Title', 'HSV Image', 'FontSize', 12, ...
                             'BackgroundColor', 'white', ...
                             'Position', [0.75 0.55 0.2 0.4]);
    hAxesHSVImage = axes('Parent', hPanelHSVImage, 'Position', [0.1, 0.1, 0.8, 0.8]);
    
    % Panel for GLCM features
    hPanelGLCM = uipanel('Title', 'GLCM Features', 'FontSize', 12, ...
                         'BackgroundColor', 'white', ...
                         'Position', [0.55 0.05 0.4 0.4]);
    hTableGLCM = uitable('Parent', hPanelGLCM, 'Position', [20, 20, 350, 150], ...
                         'ColumnName', {'0°', '45°', '90°', '135°'}, ...
                         'RowName', {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    % Panel for probabilities
    hPanelProbs = uipanel('Title', 'Predicted Labels with Probabilities', 'FontSize', 12, ...
                          'BackgroundColor', 'white', ...
                          'Position', [0.05 0.05 0.4 0.4]);
    hTextProbabilities = uicontrol('Style', 'text', 'Parent', hPanelProbs, 'Position', [10, 10, 300, 250], ...
                                   'HorizontalAlignment', 'left', 'FontSize', 12, ...
                                   'BackgroundColor', 'white', 'FontName', 'Arial');

    % Button to input image
    uicontrol('Style', 'pushbutton', 'String', 'Input Image', ...
              'Position', [600, 650, 100, 30], 'Callback', @inputImageCallback);
          
    function inputImageCallback(~, ~)
        % Load the trained models
        modelFilePath = '/Users/putragandadewata/Desktop/PCD_20Jun/acne4/knnMultiLabelModels.mat'; % Adjust path accordingly
        csvFilePath = '/Users/putragandadewata/Desktop/PCD_20Jun/acne4/train/_classes.csv'; % Adjust path accordingly
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
        
        % Combine features (padding to make 64 features if needed)
        imgFeatures = [hsvHist, glcmFeatures];
        if length(imgFeatures) < 64
            imgFeatures = [imgFeatures, zeros(1, 64 - length(imgFeatures))];
        elseif length(imgFeatures) > 64
            imgFeatures = imgFeatures(1:64);
        end
        
        % Predict probabilities for each class
        probabilities = zeros(1, length(classNames));
        for i = 1:length(classNames)
            [~, score] = predict(binaryModels{i}, imgFeatures);
            probabilities(i) = score(2); % Probability of being class 1
        end
        
        % Display the probabilities
        [sortedProbs, sortedIdx] = sort(probabilities, 'descend');
        probText = 'Predicted labels with probabilities:';
        for i = 1:length(classNames)
            probText = [probText, sprintf('\n %s: %.2f%% ', classNames{sortedIdx(i)}, sortedProbs(i) * 100)];
        end
        set(hTextProbabilities, 'String', probText);
        
        % Display HSV scatter plot
        [h, s, v] = extractHSVValues(img); % New function to get HSV values
        scatter3(hAxesHSV, h, s, v, 5, [h, s, v], 'filled');
        xlabel(hAxesHSV, 'Hue');
        ylabel(hAxesHSV, 'Saturation');
        zlabel(hAxesHSV, 'Value');
        title(hAxesHSV, 'HSV Scatter Plot');
        
        % Display GLCM features in table
        glcmTableData = num2cell(reshape(glcmFeatures, 4, 4)');
        set(hTableGLCM, 'Data', glcmTableData);
        
        % Display HSV image with highlighted acne areas
        hsvImage = highlightAcneAreas(img);
        imshow(hsvImage, 'Parent', hAxesHSVImage);
        title(hAxesHSVImage, 'HSV Image with Highlighted Acne Areas');
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

function [h, s, v] = extractHSVValues(img)
    % Convert image to HSV
    hsvImg = rgb2hsv(img);
    
    % Reshape HSV channels into vectors
    h = reshape(hsvImg(:,:,1), [], 1);
    s = reshape(hsvImg(:,:,2), [], 1);
    v = reshape(hsvImg(:,:,3), [], 1);
end

function hsvImage = highlightAcneAreas(img)
    % Convert image to HSV
    hsvImg = rgb2hsv(img);
    
    % Define thresholds for highlighting acne areas
    hueThresh = [0, 0.1]; % Example values, adjust based on your criteria
    saturationThresh = [0.6, 1]; % Example values, adjust based on your criteria
    valueThresh = [0.4, 1]; % Example values, adjust based on your criteria
    
    % Create a mask for acne areas based on HSV thresholds
    acneMask = (hsvImg(:,:,1) >= hueThresh(1) & hsvImg(:,:,1) <= hueThresh(2)) & ...
               (hsvImg(:,:,2) >= saturationThresh(1) & hsvImg(:,:,2) <= saturationThresh(2)) & ...
               (hsvImg(:,:,3) >= valueThresh(1) & hsvImg(:,:,3) <= valueThresh(2));
    
    % Convert back to RGB to highlight
    rgbImg = hsv2rgb(hsvImg);
    
    % Invert colors in the masked area
    invertedRGBImg = rgbImg;
    invertedRGBImg(:,:,1) = 1 - invertedRGBImg(:,:,1); % Invert Red channel
    invertedRGBImg(:,:,2) = 1 - invertedRGBImg(:,:,2); % Invert Green channel
    invertedRGBImg(:,:,3) = 1 - invertedRGBImg(:,:,3); % Invert Blue channel
    
    % Apply the inverted colors to the acne areas
    for channel = 1:3
        rgbImg(:,:,channel) = rgbImg(:,:,channel) .* ~acneMask + invertedRGBImg(:,:,channel) .* acneMask;
    end
    
    % Convert to HSV image for display
    hsvImage = rgb2hsv(rgbImg);
end
