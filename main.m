%% Loading in dataset
% Training Set
 facesfolder = 'C:\Users\sumig\OneDrive\Documents\MATLAB\EC520 HW\Face ID Project\ResizedTrain\'; 
 OutputFolder = 'C:\Users\sumig\OneDrive\Documents\MATLAB\EC520 HW\Face ID Project\PreprocecessedTrain\';  % Set as needed [EDITED]
 dinfo = dir(fullfile(facesfolder, '*.png'));% image extension
 for K = 1 : length(dinfo)
   thisimage = dinfo(K).name;
   Img   = imread(thisimage);
   Gray  = im2gray(Img);
   Gray_G = im2double(Gray);
   GrayS = imresize(Gray_G, [64, 74], 'bilinear');
   imwrite(GrayS, fullfile(OutputFolder, thisimage));  % [EDITED]
 end
%%
% Test Set
facesfolder_test = 'C:\Users\sumig\OneDrive\Documents\MATLAB\EC520 HW\Face ID Project\ResizedTest\';
OutputFolder_test = 'C:\Users\sumig\OneDrive\Documents\MATLAB\EC520 HW\Face ID Project\Test\';

tinfo = dir(fullfile(facesfolder_test, '*.png')); 
for J = 1 : length(tinfo)
  thisimage_test = tinfo(J).name;
  Img_test   = imread(thisimage_test);
  Gray_test  = im2gray(Img_test);
  Gray_G_test = im2double(Gray_test);
  GrayS_test = imresize(Gray_G_test, [64, 74], 'bilinear');
  imwrite(GrayS_test, fullfile(OutputFolder_test, thisimage_test));  % [EDITED]
end

%% Preprocessing Data into single matrix where each row contains the image from training and test data
[images_train] = read_face_images; 
[images_test] = read_face_images_test;
 
norm_images_train = images_train/255;

[m,n] = size(images_train);
[p,q] = size(images_test);
%% Creating Labels (6 Images for Training; 3 for Testing)

numSubjects = 15;
labels = 1:numSubjects;
lTrain = repmat(labels, 6, 1);
lTrain = reshape(lTrain, 6*numSubjects, 1);
lTest = repmat(labels, 3, 1);
lTest = reshape(lTest, 3*numSubjects, 1);

%% TRAINING:
 % Determining the mean of the face image
 figure()
 subplot(1,2,1)
 mean_image_train = mean(images_train);
 imshow(reshape(mean_image_train,[64,74]), []);
 title('Mean Face - Unnormalized')
 
 subplot(1,2,2)
 normalized_mean_image_train = mean_image_train/m;
 imshow(reshape(normalized_mean_image_train, [64, 74]), []);
 title('Mean Face - Normalized')
 
 % Finding the difference between each image and the mean face
 diff_train = images_train - mean_image_train;
 
 % Computing covariance from diff
 covmat_train = (1/m)*(diff_train'*diff_train);
 
 % V represents a matrix whose columns are eigenvectors
 % D represents a diagonal matrix with eigenvalues across the diagonal
 [V, D] = eig(covmat_train);
 
 % Sort the eigenvalues in descending order
 k = 6;
 [d,ind] = sort(diag(D), 'descend');
 top_k_eigenvalues_train = D(ind(1:k),ind(1:k));
 top_k_eigenvectors_train = V(:,ind(1:k));
 
 % The corresponding matrix V has rows that are
 % EIGENFACES of training dataset
 
 % kvalues = [10 25 50 100 250 500 750 1000]; 
 % numkvalues = length(kvalues);
 
 % Reshaping the top k eigenvectors into eigenfaces:
  figure()
  for i = 1:k
      subplot(2, 3, i);
      eigenface = reshape(top_k_eigenvectors_train(:,i), [64,74]); % Eigenfaces are column vectors
      imshow(eigenface, []);
      title(sprintf('Eigenface %d', i))
  end
 
  sgtitle('Top K Eigenfaces')
  
 % Here, we want to compute the weights of each eigenvector -> product of
 % eigenvector and the difference between image vector and mean image
 % Use the top k eigenvectors:
 
 w_k = size(1,k); % Weights are put into 1 row vector of size k
 
 for i = 1:k
     for j = 1:m
        w_k(j,i) = diff_train(j,:)*top_k_eigenvectors_train(:,i);
     end
 end
 
 %% Finding Euclidean Distance for Each Test Image
%  w_k_new = size(1,k);
%  
%  for i = 1:p
%      thisimage = tinfo(i).name;
%      new_face = imread(thisimage);
%      new_face = reshape(new_face, [1, q]);
%      new_diff = new_face - mean_image_train;
%      w_k_new = new_diff*top_k_eigenvectors;
%  end
% 
%  s = zeros(1,q);
% 
%  for i = 1:p
%      for j = 1:k
%          s(j) = (w_k_new(j) - w_k(i,j))^2;
%      end
%      
%      eu_dist(i) = sum(s);
%  end


accuracy = runTest(norm_images_train', images_test, mean_image_train, top_k_eigenvectors_train, k, lTrain, lTest);

% new_face = double(imread('David_Wells_0003.pgm'));
% new_face = reshape(new_face, [1, 4096]);
% 
% new_diff = new_face - mean_image;
% 
% w_k_new = new_diff*top_k_eigenvectors;
% 

% 
% 
% %for i = 1:numkvalues
%     %k = kvalues(j);
%     %k = 10
% %end
% 
% % Perform Reconstruction
% % 
% % for i = 1:m
% %     reconstructed_face_vector(:,i) = mean_image + eigen_vectors_norm(:,i)*w_k(:,i)';
% % end
% 
% % Append Image Weights
% 
% % Perform RMS (root mean square)
% 
