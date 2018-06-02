function [U S] = Image_Space(X_train,m,n)
avg_face = sum(X_train)/size(X_train,1);
X_train = bsxfun(@minus, X_train, avg_face);
Co_Var = (1/m)*(X_train)'*(X_train);
[U, S, V] = svd(Co_Var);
end