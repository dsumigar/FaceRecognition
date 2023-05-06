%This function takes in an n x 4096 data matrix X and an index i. 
%It extracts the ith row of X and displays it as a grayscale 64 x 64 image.

function show_subject_image(X, labels)

    if (nargin < 2)
        disp('No index selected. Displaying first face image.')
        i = 1;

        figure()
        colormap('gray')
        shg
        axis off
        imagesc([reshape(X(i, :), 64, 74)]);
    else
        unique_labels = unique(labels);
        num_Subjects = length(unique_labels);
        
        subplot_rows = floor(sqrt(num_Subjects));
        subplot_cols = ceil(num_Subjects/subplot_rows);
    
        figure()
        colormap('gray')
        shg
    
        for i = 1:1:num_Subjects
            subject_index = find(labels == i);
            subplot(subplot_rows, subplot_cols, i)
            %imagesc([reshape(X(subject_index(1), :), 74, 64)]);
            imshow(reshape(X(subject_index(1), :), [64, 74]), [])
            axis off
            title(sprintf('Subject %f'), unique_labels(i))
        end
    end

    % sgtitle('Training Subjects')
    
end