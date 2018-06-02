function [T  W avg_face] = Image_Space(X_train,m,n)
avg_face = sum(X_train)/size(X_train,1);
fprintf('The average face is...\n');
figure; displayData(avg_face);
X_train = bsxfun(@minus, X_train, avg_face);
Co_Var = (X_train)'*(X_train);
[U, S, V] = svd(Co_Var);
K = trace(S);
l = 0;
k = 0;
for i = 1:n
    l = l + 1;
    k = trace(S(1:i,1:i));
    if k/K >= 0.90
        break;
    end
end
l
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
E = U(:, (1:l));
fprintf('\n The first ten eigenfaces are...\n');
figure; displayData(E(:, (1:10))');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
T = E*E';
fprintf('\n The projection of first five faces on FACESPACE is...\n');
figure; subplot(1,2,1);
displayData(X_train(235:240,:));
subplot(1,2,2);
displayData(X_train(235:240,:)*T');
W = X_train*E;
end


    

