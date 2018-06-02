%In this code segment, we do not train our neural network.
%We just use the value of the optimal parameters obtained 
%from Analysis.m to get the final accuracy

load('project.mat')                                     %loads training,crossvalidation and test sets
load('eigenvalues.mat')                                 %loads the eigenvector and eigenvalues matrix obtained beforehand from the covariance matrix
load('eigenvectors.mat')
load('PARAMS.mat')                                      %loads the structure obtained from Analysis.m
Theta1 = PARAMETERS(143).theta1;                        
Theta2 = PARAMETERS(143).theta2;
features = PARAMETERS(143).features;                    %get the value of optimal parameters from the structure "PARAMETERS"
lambda = PARAMETERS(143).lambda;
avg_face = sum(x)/size(x,1);
fprintf('The average face is...\n');
figure; displayData(avg_face);
x = bsxfun(@minus, x, avg_face);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
E = U(:, (1:features));
fprintf('\n The first ten eigenfaces are...\n');
figure; displayData(E(:, (1:12))');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
T = E*E';
r = randperm(240);
fprintf('\n The projection of any five faces on FACESPACE is...\n');
figure; subplot(1,2,1);
displayData(x((r(1:6)),:));
subplot(1,2,2);
displayData(x((r(1:6)),:)*T');
W = x*E;
pred = predict(Theta1, Theta2, W);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
x_cv = bsxfun(@minus, x_cv, avg_face);
r = randperm(120);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('\n The projection of any five cross validation faces on FACESPACE is...\n');
figure; subplot(1,2,1);
displayData(x_cv((r(1:6)),:));
subplot(1,2,2);
displayData(x_cv((r(1:6)),:)*T');
W = x_cv*E;
pred = predict(Theta1, Theta2, W);
cv_acc = mean(double(pred == y_cv)) * 100;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('\nCrossvalidation Set Accuracy: %f\n', cv_acc);
x_test = bsxfun(@minus, x_test, avg_face);
W = x_test*E;
pred = predict(Theta1, Theta2, W);                              %predictions for test set
t_acc = mean(double(pred == y_test)) * 100;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('test Set Accuracy: %f\n', t_acc);
[x_norm1, ~, ~] = featureNormalize(x_test);
image_norm = x_norm1(input('enter a number less than 40: '),:);
figure;
subplot(1,2,1);
displayData(image_norm);
title('Input Image')
image_red = projectData(image_norm, U, features);
prediction = predict(Theta1, Theta2, image_red);
fprintf('image belongs to face class: %d\n', prediction);
subplot(1,2,2);
displayData(x(((prediction-1)*(6))+(1:6), :))
title('Input Image belongs to this class')



    

