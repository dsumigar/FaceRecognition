% This function reads in all images of the Face ID dataset

function [images] = read_face_images

dim = 64; % Choosing dimensions of 64 x 64


image_files = dir('lfw1000_training.zip');
numOfFiles = length(image_files);
sumOfImages = double(zeroes(dim,dim)); 

for i = 1:numOfFiles
    current_file_name = image_files(i).name;
    
    initial_file = imread(current_file_name); % Want to read the current image
    [rows, columns, numOfColorChannels] = size(initial_file);
    
    % Want to grayscale every input image (if the input is in RGB format):
    if numOfColorChannels > 1
        current_image = rgb2gray(initial_file);
    else
        current_image = initial_file;
    end
    
    % Resizing our image:
    current_image = imresize(double(current_image), [dim dim]);
    [m, n] = size(current_image); % Assign dimension of image
    images(:, i) = {reshape(current_image,[m*n,1])}; % Creating a matrix of all images, where each column is the image (Y in their case)
       
end

end
