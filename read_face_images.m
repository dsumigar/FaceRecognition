% This function reads in all images in the folder 'faceFolder'. 
% Each m x n image is reshaped into a length-(m * n) row vector. 
% These row vectors are stacked on top of one another to get one data
% matrix, each with (m * n) columns. The matrix consists of all
% the face images as row vectors.

function [images] = read_face_images(faceFolder, imgExt)

    faceFiles = dir(fullfile(faceFolder, imgExt)); 
    faceFileNames = {faceFiles.name};
    numOfFaces = length(faceFileNames);

    [m n] = size(imread(strcat(faceFolder, faceFileNames{1})));            % Each image should've already been resized to the same dimensions

    images = [];

    for i = 1:numOfFaces
        faceFileNameTemp = faceFileNames{i};
        faceImageTemp = imread(strcat(faceFolder, faceFileNameTemp));
        faceVecTemp = reshape(faceImageTemp, 1, m * n);                    % Resize image into stretched out column vector
        images = [images; faceVecTemp];
    end
    
    images = double(images);

end
