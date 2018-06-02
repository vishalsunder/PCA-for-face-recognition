load('project.mat')
train_size = size(x,1);
cv_size = size(x_cv,1);
test_size = size(x_test,1);
pred = zeros(cv_size,1);
k = zeros(40,2500);
for i = 1:40
    k(i,:) = sum(x((6*i-5:6*i),:))/6;
end
pred = zeros(cv_size,1);
for i = 1:cv_size
    I = zeros(40,1);
    for j = 1:40
        I(j) = norm(x_cv(i,:)-k(j,:));
    end
    pred(i) = find(I==min(I));
end
cv_acc = mean(double(pred == y_cv)) * 100
