%https://blog.cordiner.net/2010/12/02/eigenfaces-face-recognition-matlab/
clear all;
close all;
input_dir = 'C:\Users\ASI I\Desktop\vinayak sable\eigenfaces\vinayak sable\training set';
image_dims = [112,92];
 
filenames = dir(fullfile(input_dir, '*.png'));
num_images = numel(filenames);
images = [];
for n = 1:num_images
    filename = fullfile(input_dir, filenames(n).name);
    img = imread(filename);
    if n == 1
        images = zeros(prod(image_dims), num_images);
    end
    images(:, n) = img(:);
end



% steps 1 and 2: find the mean image and the mean-shifted input images
mean_face = mean(images, 2);
shifted_images = images - repmat(mean_face, 1, num_images);



 
% steps 3 and 4: calculate the ordered eigenvectors and eigenvalues
[evectors, score, evalues] = pca(images');
 
% step 5: only retain the top 'num_eigenfaces' eigenvectors (i.e. the principal components)
num_eigenfaces = 12;
evectors = evectors(:, 1:num_eigenfaces);
 

% step 6: project the images into the subspace to generate the feature vectors
features = evectors' * shifted_images;

input_image=imread('C:\Users\ASI I\Desktop\vinayak sable\eigenfaces\vinayak sable\testing set\4.png');


% calculate the similarity of the input to each training image
 feature_vec = evectors' * (double([input_image(:)]) - mean_face);
similarity_score = arrayfun(@(n) 1 / (1 + norm(features(:,n) - feature_vec)), 1:num_images);
 
% find the image with the highest similarity
[match_score, match_ix] = max(similarity_score);
 
% display the result
figure, imshow([input_image reshape(images(:,match_ix), image_dims)]);
title(sprintf('matches %s, score %f', filenames(match_ix).name, match_score));

% display the eigenvalues
normalised_evalues = evalues / sum(evalues);
figure, plot(cumsum(normalised_evalues));
xlabel('No. of eigenvectors'), ylabel('Variance accounted for');
xlim([1 30]), ylim([0 1]), grid on;





