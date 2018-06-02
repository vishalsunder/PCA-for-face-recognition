close all; clear; clc;
load('project.mat')
clear x_cv x_test y y_cv y_test ;
load('RecognitionData.mat')
[x_norm1, ~, ~] = featureNormalize(x_test);
image_norm = x_norm1(input('enter a number less than 40: '),:);
figure;
subplot(1,2,1);
displayData(image_norm);
title('Input Image')
image_red = projectData(image_norm, U, k);
prediction = predict(Theta1, Theta2, image_red);
fprintf('image belongs to face class: %d\n', prediction);
subplot(1,2,2);
displayData(x(((prediction-1)*(6))+(1:6), :))
title('Input Image belongs to this class')