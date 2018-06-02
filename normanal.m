load('eigenvectors.mat')
load('project.mat')
train_size = size(x,1);
cv_size = size(x_cv,1);
test_size = size(x_test,1);
avg_face = sum(x)/size(x,1);
class = 40;

x = bsxfun(@minus, x, avg_face);
x_cv = bsxfun(@minus, x_cv, avg_face);
cv_acc = zeros(197,1)
n = 0;
for l = 20:5:1000
    n = n+1;
    n
E = U(:, (1:l));                                          %creates a matrix of first 169 (or as many as you like) eigenvectors which are used to define the
                           
W = x*E;                    %we calculate the weights of each of the "Principle Components", hence the no. of features describing the image is greatly reduced

w = x_cv*E;
k = zeros(40,l);
for i = 1:40
    k(i,:) = sum(W((6*i-5:6*i),:))/6;
end
pred = zeros(cv_size,1);
for i = 1:cv_size
    I = zeros(40,1);
    for j = 1:40
        I(j) = norm(w(i,:)-k(j,:));
    end
    pred(i) = find(I==min(I));
end
cv_acc(n) = mean(double(pred == y_cv)) * 100;
end
