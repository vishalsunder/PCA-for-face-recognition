function [I_train] = Get_Images(I_train,m)
SrcFiles = dir('E:\machine learning\Face detection\training_images\s24\*.pgm');
I = zeros(1,m*m);
for i = 1:length(SrcFiles)
    im = imread(SrcFiles(i).name);
    im = mat2gray(im);
    im = imresize(im,[m,m]);
    im = im(:)';
    I = [I;im];
end
I_train = [I_train;I(2:end,:)];
end
