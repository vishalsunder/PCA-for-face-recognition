load('project.mat')                           %loads training,crossvalidation and test sets
load('eigenvalues.mat')                       %loads the eigenvector and eigenvalues matrix obtained beforehand from the covariance matrix
load('eigenvectors.mat')
avg_face = sum(x)/size(x,1);
fprintf('The average face is...\n');
figure; displayData(avg_face);
x = bsxfun(@minus, x, avg_face);
r = 0;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
E = U(:, (1:60));                                          %creates a matrix of first 170 (or as many as you like) eigenvectors which are used to define the
fprintf('\n The first ten eigenfaces are...\n');            %face space with 170 (or as many as you like) "Principle Components"
figure; displayData(E(:, (1:12))');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
T = E*E';                 %this is a "Transformation Matrix" which defines a linear transformation which is the projection of an image onto the "face space"
r = randperm(240);                                          
fprintf('\n The projection of any five faces on FACESPACE is...\n');
figure; subplot(1,2,1);
displayData(x((r(1:6)),:));
subplot(1,2,2);
displayData(x((r(1:6)),:)*T');     %we randomly pick any of the five images from our training set and print its projection onto the "Face Space"                           
W1 = x*E;                    %we calculate the weights of each of the "Principle Components", hence the no. of features describing the image is greatly reduced
input_layer_size  = 60; 
hidden_layer_size = 150;   %FOR NEURAL NETWORK
num_labels = 40; 
fprintf('\nProgram paused. Press enter to start training the ANN.\n');
pause;
[Theta1,Theta2] = Training(input_layer_size, hidden_layer_size,num_labels,W1,y,12);     % TRAINING OF NEURAL NETWORK
fprintf('\nProgram paused. Press enter to see accuracy on training set.\n');
pause;
pred = predict(Theta1, Theta2, W1);     %predictions for training data
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
x_cv = bsxfun(@minus, x_cv, avg_face);
r = randperm(120);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('\n The projection of any five cross validation faces on FACESPACE is...\n');
figure; subplot(1,2,1);
displayData(x_cv((r(1:6)),:));
subplot(1,2,2);
displayData(x_cv((r(1:6)),:)*T');     %we randomly pick any of the five images from our cross validation set and print its projection onto the "Face Space" 
W = x_cv*E;
pred = predict(Theta1, Theta2, W);     %predictions for cross validation data
cv_acc = mean(double(pred == y_cv)) * 100;
fprintf('\nCrossvalidation Set Accuracy: %f\n', cv_acc);



    

