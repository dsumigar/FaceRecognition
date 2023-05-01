clear Workspace
clear all
close all
clc

%% Load in Training Dataset and Create Labels
facesFolder_train = 'C:\Users\sumig\OneDrive\Documents\MATLAB\EC520 HW\Face ID Project\ResizedTrain\';            % Location of training set of face images
outputFolder_train = 'C:\Users\sumig\OneDrive\Documents\MATLAB\EC520 HW\Face ID Project\PTrain\';    % Initialize folder to hold pre-processed training set of face images
dinfo = dir(fullfile(facesFolder_train, '*.png'));                                                                    % Image extension

imgExt = '*.png';

num_train_images = length(dinfo);                                                                                     % Number of training images in folder
lTrain = zeros(num_train_images, 1);                                                                                  % Initialize training labels vector

% Choose a random image to get the dimensions of from the dataset and
% resize all images to those dimensions => but training set images should be all of the sizes
random_index = randi([1, num_train_images], 1, 1);
[a b] = size(im2gray(im2double(imread(dinfo(random_index).name))));

for i = 1 : num_train_images                                                                                            % Iterate through each face image file
  thisImage_train = dinfo(i).name;                                                                                    % Obtain file name for i-th face image
  Img_train   = imread(thisImage_train);                                                                              % Read i-th file as image
  Gray_train  = im2gray(Img_train);                                                                                   % Convert image to grayscale
  Gray_G_train = im2double(Gray_train);                                                                               % Make sure that the face image matrix is of type 'double' => will allow us to perform calculations more easily later on
  GrayS_train = imresize(Gray_G_train, [b, a], 'bilinear');                                                         % Resize i-th face image so that they are all of the same height and width
  imwrite(GrayS_train, fullfile(outputFolder_train, thisImage_train));                                                % Save resized & grayscaled face image in pre-processed training set of face images folder

  for j = 1 : num_train_images                                                                                        % Create labels based on file names with format of 'subject##.*.png'
      if (contains(dinfo(i).name, strcat('subject0', num2str(j))))
          lTrain(i) = j;
      elseif (contains(dinfo(i).name, strcat('subject', num2str(j))))
          lTrain(i) = j;
      end
  end
end

%% Load in Testing Dataset
facesFolder_test = 'C:\Users\sumig\OneDrive\Documents\MATLAB\EC520 HW\Face ID Project\ResizedTest\';
outputFolder_test = 'C:\Users\sumig\OneDrive\Documents\MATLAB\EC520 HW\Face ID Project\PTest\';
tinfo = dir(fullfile(facesFolder_test, '*.png')); 

num_test_images = length(tinfo);
lTest = zeros(num_test_images, 1); 

for i = 1 : num_test_images
  thisImage_test = tinfo(i).name;
  Img_test   = imread(thisImage_test);
  Gray_test  = im2gray(Img_test);
  Gray_G_test = im2double(Gray_test);
  GrayS_test = imresize(Gray_G_test, [b, a], 'bilinear');
  imwrite(GrayS_test, fullfile(outputFolder_test, thisImage_test));

  for j = 1 : num_test_images                                                                                        
      if (contains(tinfo(i).name, strcat('subject0', num2str(j))))
          lTest(i) = j;
      elseif (contains(tinfo(i).name, strcat('subject', num2str(j))))
          lTest(i) = j;
      end
  end
end

%% Convert Pre-Processed Face Image Data into Single Matrix
% Each row contains the image information of 1 face
[images_train] = read_face_images(outputFolder_train, imgExt); 
[images_test] = read_face_images(outputFolder_test, imgExt);
 
norm_images_train = images_train/255;                                                                                 % Normalize the training images matrix

[m, n] = size(images_train);
[p, q] = size(images_test);

%% Computations for PCA
% Determining the Mean of the Face Image
figure()
subplot(1,2,1)
mean_image_train = mean(images_train);
imshow(reshape(mean_image_train, [b, a]), []);
title('Mean Face - Unnormalized')
 
subplot(1,2,2)
normalized_mean_image_train = mean_image_train/m;
imshow(reshape(normalized_mean_image_train, [b, a]), []);
title('Mean Face - Normalized')

%% Finding the Difference Between Each Image and the Mean Face
diff_train = images_train - mean_image_train;

% Computing Covariance from the Difference
covmat_train = (1/m) * (diff_train'*diff_train);

% V represents a matrix whose columns are eigenvectors
% D represents a diagonal matrix with eigenvalues across the diagonal
[V, D] = eig(covmat_train);

%% Principal Component Analysis (PCA)
% Dimensions to try for PCA dimensionality reduction
kvalues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 90];
kfolds = length(kvalues);

percentage_correct = zeros(1, kfolds);

% Testing different numbers of top k eigenvectors to consider
for r = 1:kfolds
  % Sort the eigenvalues in descending order
  k = kvalues(r);
  [d, ind] = sort(diag(D), 'descend');
  top_k_eigenvalues_train = D(ind(1:k), ind(1:k));
  top_k_eigenvectors_train = V(:, ind(1:k));
 
  % Here, we want to compute the weights of each eigenvector -> product of
  % eigenvector and the difference between image vector and mean image
  % Use the top k eigenvectors:
 
  w_k = size(1, k);                                                                                                   % Weights are put into 1 row vector of size k
 
  for i = 1:k
      for j = 1:m
         w_k(j, i) = diff_train(j, :) * top_k_eigenvectors_train(:, i);
      end
  end
 
%% Finding Euclidean Distance for Each Test Image
  new_diff = images_test - mean_image_train;
  w_k_new = new_diff * top_k_eigenvectors_train;

  s = zeros(1, q);
  pred = zeros(1, p);
  eu_dist = zeros(p, num_train_images);

  for i = 1:num_test_images
      for j = 1:num_train_images
          for l = 1:k
              s(j, l) = (w_k_new(i, l) - w_k(j, l))^2;
          end

          eu_dist(i, j) = sum(s(j, :));
      end
    
      if (length(find(eu_dist(i, :) == min(eu_dist(i, :)))) > 1)
          pred(i) = randi([1, length((find(eu_dist(i, :) == min(eu_dist(i, :)))))]);
      else
          pred(i) = find(eu_dist(i, :) == min(eu_dist(i, :)));
      end
    
  end

  correct = 0;

  for i = 1:p
      if (lTest(i) == lTrain(pred(i)))
          correct = correct + 1;
      end
  end

  percentage_correct(r) = correct/p * 100;
  
  %% k-NN Classification for PCA
[M, ~] = size(images_train);
[P, ~] = size(images_test);

K = 3;
s = zeros(1, q);
pred_knn_PCA = zeros(1, p);

[n_rows, n_cols] = size(eu_dist);

for i = 1:n_rows
    [top_max, ind] = mink(eu_dist(i,:),K);
    guess = lTrain(ind);
    pred_knn_PCA(i) = mode(guess);
end

correct = 0;

for i = 1:P
  if (lTest(i) == pred_knn_PCA(i))
      correct = correct + 1;
  end
end

kNNPCA_percentage_correct(r) = correct/p * 100
end


%% Plot Results
 figure()
 plot(kvalues, percentage_correct)
 title('Percentage Correct vs Top k Number of Eigenfaces Considered')
 ylabel('Percentage Correct (%)')
 xlabel('Top k Number of Eigenfaces')

%% Obtain Best k Number of Eigenvectors to Consider
% Assuming at least 95% accurate is 'good enough' => find lowest k-value
% that fits this criteria
percentage_best = find(percentage_correct > 84);
k_best =  kvalues(percentage_best(1));

%% Initialization and Setup of Workspace to Perform Face Recognition - LDA
% Load in Dataset and Separate Face Information into Respective Classes (Persons)
[LDA_images_train] = read_face_images(outputFolder_train, imgExt); 
[LDA_images_test] = read_face_images(outputFolder_test, imgExt);

[LDA_num_train_images, LDA_train_dimensions] = size(LDA_images_train);
[LDA_num_test_images, LDA_test_dimensions] = size(LDA_images_test);

Subjects = unique(lTrain);
Separate_Subjects = zeros(length(Subjects), [], LDA_train_dimensions);

for i = 1:LDA_num_train_images
    for j = 1:length(Subjects)
        if lTrain(i) == Subjects(j)
            Separate_Subjects(j, end + 1, :) = LDA_images_train(i, :);
        end
    end
end

%% Find Mean Face of Each Class (Person)
avgFace_Classes = zeros(length(Subjects), LDA_train_dimensions);
for i = 1:length(Subjects)
    avgFace_Classes(i, :) = mean(Separate_Subjects(Subjects(i), :, :));
end

% Normalize to range of [0, 1]
avgFace_Classes = avgFace_Classes / 255;
LDA_images_train = LDA_images_train / 255;

% Calculating the Deviation of Each Image to its Class (Person)
LDA_diff = zeros(LDA_num_train_images, LDA_train_dimensions);

for i = 1:LDA_num_train_images
    for j = 1:length(Subjects)
        if lTrain(i) == Subjects(j)
            LDA_diff(i, :) = LDA_images_train(i, :) - avgFace_Classes(j, :);
        end
    end
end

%% Perform PCA and Project Images onto Lower Dimensional Eigenspace as Setup for LDA => To Solve the Problem of invertibility of S_w (Computer Scattering Matrices in Reduced Dimensions)
L = LDA_diff' * LDA_diff;                                           % L is the surrograte of covariance matrix C = A * A'

[LDA_V, LDA_D] = eig(L);                                            % Diagonal elements of D are the eigenvalues for both L = A' * A and C = A * A'                

k = k_best;

% Sorting and eliminating small eigenvalues
[LDA_d, LDA_ind] = sort(diag(LDA_D), 'descend');
LDA_top_k_eigenvalues_train = LDA_D(LDA_ind(1:k), LDA_ind(1:k));
LDA_top_k_eigenvectors_train = LDA_V(:, LDA_ind(1:k));

% Here, we want to compute the weights of each eigenvector -> product of
% eigenvector and the difference between image vector and mean image
% Use the top k eigenvectors:

LDA_w_k = size(1, k);                                                                                                 % Weights are put into 1 row vector of size k

for i = 1:k
  for j = 1:LDA_num_train_images
     LDA_w_k(j, i) = LDA_diff(j, :) * LDA_top_k_eigenvectors_train(:, i);
  end
end

% Project centered image vectors onto eigenspace
ProjectedImage_PCA = LDA_w_k' * LDA_diff;

%% Linear Discriminant Analysis (LDA)
% The problem with doing LDA is that SW is not invertible. So instead
% compute the scattering matrices in reduced dimension . . .

% For scatter matrices that are singular, we project the image set to a
% lower dimensional space so that the scatter matrices become non-singular

% Computing Scatter Matrices: 
% First, use PCA to reduce the feature space dimension to N - c (discarding
% smallest c - 1 eigenvectors)

% Sort the eigenvalues to keep the top N - c eigenvectors
%reduced_dim = m - length(Subjects);
reduced_dim = 30;
PCA_on_LDA = covmat_train;
[V_PCA_LDA, D_PCA_LDA] = eig(PCA_on_LDA);
[LDA_d_N_minus_c, LDA_ind_N_minus_c] = sort(diag(D_PCA_LDA), 'descend');
LDA_top_N_minus_c_eigenvalues = D_PCA_LDA(LDA_ind_N_minus_c(1:reduced_dim), LDA_ind_N_minus_c(1:reduced_dim));
LDA_top_N_minus_c_eigenvectors = V_PCA_LDA(:, LDA_ind_N_minus_c(1:reduced_dim));

%%
LDA_basis_reduced = LDA_top_N_minus_c_eigenvectors;             % Want to take the first m - k eigenvectors (reduce the basis)

%LDA_basis_reduced = diff_train * LDA_basis_reduced;            % Project the the training images onto the reduced basis (Wpca)
PCA_LDA_train = diff_train * LDA_basis_reduced;                 % Compute projection matrix (n by m - k)
mean_PCA_LDA_train = mean(PCA_LDA_train,1);                     % Compute (Wpca'ubar)

% Allocating space for scatter matrices:
S_b = zeros(reduced_dim, reduced_dim);
S_w = zeros(reduced_dim, reduced_dim);

for i = 1 : length(Subjects)
    classes(i) = sum(lTrain==i);                                % classes stores number of samples of class i (6)
    mean_class(i,:) = mean(PCA_LDA_train(lTrain==i,:),1);       % Compute u_i
    temp_S_b = mean_class(i,:) - mean_PCA_LDA_train;            % Determine the difference between the mean of each class and the overall mean
    S_b = S_b + classes(i) * (temp_S_b'*temp_S_b);              % Between class scatter
    S_w = S_w + classes(i) * cov(PCA_LDA_train(lTrain==i,:));   % Within class scatter
end

%% Repeat Computation in Lower Dimension (c - 1)
% Now we want to further reduce the dimension to c - 1:
reduced_dim_2 = length(Subjects) - 1;
[V_scatter_reduced_dim, D_scatter_reduced_dim] = eig(S_b, S_w);                                           % Find eigenvectors of scatter matrices (forming the basis of Wfld)
[d_reduced_dim, ind_reduced_dim] = sort(diag(D_scatter_reduced_dim), 'descend');
reduced_dim_top_c_minus_1_eigenvalues = D_scatter_reduced_dim(ind_reduced_dim(1:reduced_dim_2), ind_reduced_dim(1:reduced_dim_2));
reduced_dim_top_c_minus_1_eigenvectors = V_scatter_reduced_dim(:, ind_reduced_dim(1:reduced_dim_2));

%%
% Repeat the previous computations in the reduced dimension:
Wfld = reduced_dim_top_c_minus_1_eigenvectors;                                      % Further reduction of basis to c - 1 -> Wfld
u = PCA_LDA_train * Wfld;                                                           % Product of Wfld and Wpca = Wopt        
%u = diff_train' * Wfld_Wpca;                                                        % u = Wfld*Wpca*x (Projection onto data to maximize class separation)           

%% Finding Euclidean Distance for Each Test Image
u_new = new_diff * LDA_basis_reduced * Wfld;

s = zeros(1, q);
pred = zeros(1, p);
eu_dist = zeros(p, num_train_images);

for i = 1:LDA_num_test_images
  for j = 1:LDA_num_train_images
      for l = 1:(length(Subjects) - 1)
          s(j, l) = (u_new(i, l) - u(j, l))^2;
      end

      eu_dist(i, j) = sum(s(j, :));
  end

  if (length(find(eu_dist(i, :) == min(eu_dist(i, :)))) > 1)
      aa = find(eu_dist(i, :) == min(eu_dist(i, :)));
      pred(i) = aa(randi(length(aa)));
  else
      pred(i) = find(eu_dist(i, :) == min(eu_dist(i, :)));
  end

end

correct = 0;

for i = 1:p
  if (lTest(i) == lTrain(pred(i)))
      correct = correct + 1;
  end
end

LDA_percentage_correct = correct/p * 100;

%% Testing LDA (kNN Classification)
[M, ~] = size(LDA_images_train);
[P, ~] = size(LDA_images_test);

K = 3;
s = zeros(1, q);
pred_knn = zeros(1, p);

labels = zeros(P, 1);
distances = zeros(P, K);

[n_rows, n_cols] = size(eu_dist);

for i = 1:n_rows
    [top_max, ind] = mink(eu_dist(i,:),K);
    guess = lTrain(ind);
    pred_knn(i) = mode(guess);
end

correct = 0;

for i = 1:P
  if (lTest(i) == pred_knn(i))
      correct = correct + 1;
  end
end

kNNLDA_percentage_correct = correct/p * 100
%%
reconstructed_LDA = u*reduced_dim_top_c_minus_1_eigenvectors'*LDA_basis_reduced';

for i = 1:P
    ooo(i,:) = reconstructed_LDA(i,:) + mean_image_train;
end

show_subject_image(ooo, lTest)