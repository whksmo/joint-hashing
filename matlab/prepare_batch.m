% ------------------------------------------------------------------------
function images = prepare_batch(image_files,mean_data,batch_size,local)
% ------------------------------------------------------------------------
if nargin < 2
    % load mean file
    d = load('/home/iis/Research/packages/caffe-master/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');
    mean_data = d.mean_data;
end

num_images = length(image_files);
if nargin < 3
    batch_size = num_images;
end

IMAGE_DIM = 256;
CROPPED_DIM = 227;

indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
center = floor(indices(2) / 2)+1;
num_images = length(image_files);
images = zeros(CROPPED_DIM,CROPPED_DIM,3,batch_size,'single');

parfor i=1:num_images
    % read file
    fprintf('%c Preparing %s\n',13,image_files{i});
    try
    if local
        im = imread(['.' image_files{i}]);
    else
        im = imread(image_files{i});
    end
    % resize to fixed input size
    im = single(im);
    im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
    % Transform GRAY to RGB
    if size(im,3) == 1
        im = cat(3,im,im,im);
    end

    % permute from RGB to BGR (IMAGE_MEAN is already BGR)
    im = im(:,:,[3 2 1]) - mean_data;
   % im = im(:,:,[3 2 1]);
    % Crop the center of the image
    images(:,:,:,i) = permute(im(center:center+CROPPED_DIM-1,...
    center:center+CROPPED_DIM-1,:),[2 1 3]);
    catch
        warning('Problems with file',image_files{i});
    end
end
