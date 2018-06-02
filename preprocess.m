function [im] = preprocess(image)
im = mat2gray(image);
im = imresize(im,[50,50]);
im = im(:)';
end