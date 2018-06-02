load('eigenvectors.mat')
load('project.mat')
train_size = size(x,1);
cv_size = size(x_cv,1);
test_size = size(x_test,1);
avg_face = sum(x)/size(x,1);
fprintf('The average face is...\n');
figure; displayData(avg_face);
x = bsxfun(@minus, x, avg_face);
r = 0;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
E = U(:, (1:50));                                          %creates a matrix of first 169 (or as many as you like) eigenvectors which are used to define the
fprintf('\n The first ten eigenfaces are...\n');            %face space with 169 (or as many as you like) "Principle Components"
figure; displayData(E(:, (1:10))');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
T = E*E';                 %this is a "Transformation Matrix" which defines a linear transformation which is the projection of an image onto the "face space"
r = randperm(240);                                          
fprintf('\n The projection of any five faces on FACESPACE is...\n');
figure; subplot(1,2,1);
displayData(x((r(1:6)),:));
subplot(1,2,2);
displayData(x((r(1:6)),:)*T');     %we randomly pick any of the five images from our training set and print its projection onto the "Face Space"                           
W = x*E;                    %we calculate the weights of each of the "Principle Components", hence the no. of features describing the image is greatly reduced
x_cv = bsxfun(@minus, x_cv, avg_face);
r = randperm(120);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('\n The projection of any five cross validation faces on FACESPACE is...\n');
figure; subplot(1,2,1);
displayData(x_cv((r(1:5)),:));
subplot(1,2,2);
displayData(x_cv((r(1:5)),:)*T');     %we randomly pick any of the five images from our cross validation set and print its projection onto the "Face Space" 
w = x_cv*E;
pred = zeros(cv_size,1);
for i = 1:cv_size
    I = zeros(train_size,1);
    for j = 1:240
        I(j) = norm(w(i,:)-W(j,:));
    end
    pred(i) = ceil(find(min(I))/6);
end
cv_acc = mean(double(pred == y_cv)) * 100
