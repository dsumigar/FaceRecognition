%This function reads in all images of the Face ID dataset and stretches the
%image into a column vector of dimension [4096, 1]

function [images] = read_face_images

    facesfolder = 'C:\Users\sumig\OneDrive\Documents\MATLAB\EC520 HW\Face ID Project\PreprocecessedTrain\';
    facefiles = dir(fullfile(facesfolder, '*.png')); 
    facefilenames = {facefiles.name};
    numOfFaces = length(facefilenames);

    images = [];

    for i = 1:numOfFaces
        facefilenametemp = facefilenames{i};
        faceimagetemp = imread(strcat(facesfolder,facefilenametemp));
        facevectemp = reshape(faceimagetemp,1,64 * 74); % Resize image into stretched out column vector
        images = [images; facevectemp];
    end
    
    images = double(images);

end