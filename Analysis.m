% This code segment analyses our code for different combinations of the
% number of features and value of overfitting parameter lambda and tries 
% find the combination of these two for which the cross validation accuracy
% is maximum.For every combination, we get a different set of parameter
% theta1 and theta2 which are stored in a structure. Finally, we get a
% matrix named "analysis" which stores accuracy for all different
% combinations.For every element in the matrix, there is a corresponding
% value of theta1 and theta2 which we can get manually from the
% structure.Thus, we can save the parameter for which accuracy is maximum.


load('project.mat')                                %loads training,crossvalidation and test sets
load('eigenvalues.mat')                            %loads the eigenvector and eigenvalues matrix obtained beforehand from the covariance matrix
load('eigenvectors.mat')
analysis = zeros(18,26);                    
f = 0;
a = 0;
k = 0;
PARAMETERS = struct;                              %the structure is defined
avg_face = sum(x)/size(x,1);


x = bsxfun(@minus, x, avg_face);
for features = 10:10:180
    a = a+1;
    b = 0;
    for lambda = 0:1:25
        k = k+1;
        k
        b = b+1;
        
       
        E = U(:, (1:features));
        
        
        
        
        T = E*E';
        
        
        
        
        
        W = x*E;
        input_layer_size  = features; 
        hidden_layer_size = 150;   
        num_labels = 40; 
        
        
        [Theta1,Theta2] = Training(input_layer_size, hidden_layer_size,num_labels,W,y,lambda);
        PARAMETERS(k).theta1 = Theta1;
        PARAMETERS(k).theta2 = Theta2;                                      
        PARAMETERS(k).lambda = lambda;
        PARAMETERS(k).features = features;
        
        
        
        
        x_cv1 = bsxfun(@minus, x_cv, avg_face);
        W = x_cv1*E;
        pred = predict(Theta1, Theta2, W);
        analysis(b,a) = mean(double(pred == y_cv)) * 100;
    end
end

f = find(analysis==max(max(analysis)))                               %finds the field in the structure corresponding to which maximum accuracy is obtained



    


     
