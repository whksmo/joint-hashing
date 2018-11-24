% Demo of binary codes and deep feature extraction  
% Modify 'test_file_list' and get the features of your images!

close all;
clear;

% Initialize
addpath(genpath(pwd));
fprintf('SSDH startup\n');
%%
% -----------------------------------------------------------
% 48-bits binary codes extraction
%
% input
%   	img_list.txt:  list of images files 
% output
%   	binary_codes: 48 x num_images output binary vector
%   	list_im: the corresponding image path
%
% ----- settings start here -----
% set 1 to use gpu, and 0 to use cpu
use_gpu = 1;
% binary code length
feat_len = 48;
% models
model_file = './examples/bone-finetune/solver_iter_50000.caffemodel';
model_file = './examples/cifar10/cifar10_modify_iter_50000.caffemodel';
model_file = './examples/cifar10/cifar10_48_iter_50000.caffemodel';
% model_file = './examples/cifar10/cifar10_0_iter_50000.caffemodel';


% model definition
% model_def_file = './models/SSDH/deploy.prototxt';
model_def_file = './examples/bone-finetune/deploy.prototxt';

% caffe mode setting
phase = 'test'; % run with phase test (so that dropout isn't applied)
% input data
test_file_list = './data/cifar10/demo_imgs/img_list.txt';
% ------ settings end here ------


param.nbits=24;
param.dim = 4096;
param = trainLSH(param);
% Extract binary hash codes
[feat_test, ft_fc7_tst, ipt_tst, list_im]= feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
[~, b_fc7_tst] = compressLSH(ft_fc7_tst', param);
binary_codes = (feat_test>0.5);
% binary_codes = [binary_codes; b_fc7_tst'];
save('binary48.mat','binary_codes','list_im','-v7.3');
feature.hogcolor = true;
figure(1),
set(gcf, 'Position'); 
% offset=4;   
    for i=1:12
        image = sprintf('.%s',list_im{i});
%         image = list_im(:,i);
        image = imread(image);
        hog_ft(i,:,:,:) = get_features(reshape(image,32,32,3),[]);
    %     codes = num2str(binary_codes(:,i+offset)');
    %     codes = sprintf('binary codes: %s',codes);
        subplot(4,4,i), imshow(image); 
    %     title(codes);
    end  
subplot(4,4,[13,16]), imshow(binary_codes');
%% Visualization
figure(1),
set(gcf, 'Position'); 
% offset=4;   
    for i=1:12
        image = sprintf('.%s',list_im{i});
    %     codes = num2str(binary_codes(:,i+offset)');
    %     codes = sprintf('binary codes: %s',codes);
        subplot(4,4,i), imshow(image); 
    %     title(codes);
    end  
subplot(4,4,[13,16]), imshow(binary_codes');


