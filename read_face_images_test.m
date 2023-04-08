function [images_test] = read_face_images_test

testfacesfolder = 'C:\Users\sumig\OneDrive\Documents\MATLAB\EC520 HW\Face ID Project\Test\';
testfacefiles = dir(fullfile(testfacesfolder, '*.png')); 
testfacefilenames = {testfacefiles.name};
numOfFacestest = length(testfacefilenames);

images_test = [];

    for i = 1:numOfFacestest
        testfacefilenametemp = testfacefilenames{i};
        testfaceimagetemp = imread(strcat(testfacesfolder,testfacefilenametemp));
        testfacevectemp = reshape(testfaceimagetemp,1,64 * 74); % Resize image into stretched out row vector
        images_test = [images_test; testfacevectemp];
    end
    
    images_test = double(images_test);
end