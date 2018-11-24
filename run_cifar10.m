close all;
clear;
clc
%%
% -- settings start here ---
% set 1 to use gpu, and 0 to use cpu
use_gpu = 1;

% top K returned images
top_k = 587;
feat_len = 48;%don't forget to modify the deploy.prototxt!

% models
model_file = './examples/bone-finetune/SSDH_BONE_48_iter_25000.caffemodel';
% model_file = './examples/cifar10/SSDH48_iter_50000.caffemodel';
% model_file = sprintf('./examples/cifar10/cifar10_%d_iter_50000.caffemodel',feat_len);
model_file = './examples/cifar10/cifar10_ae_iter_50000.caffemodel';

% model definition
model_def_file = './examples/bone-finetune/deploy.prototxt';
model_def_file = './examples/bone-finetune/deploy_ae.prototxt';

% train-test
use_cifar10 = true;
if(use_cifar10)
    % cifar10
    test_file_list = './data/cifar10/test-file-list.txt';
    test_label_file = './data/cifar10/test-label.txt';
    train_file_list = './data/cifar10/train-file-list.txt';
    train_label_file = './data/cifar10/train-label.txt';
else
    % bone
    test_file_list = './data/bone/test-file-list.txt';
    train_file_list = './data/bone/train-file-list.txt';
    test_label_file = './data/bone/test-label-bone.txt';
    train_label_file = './data/bone/train-label-bone.txt';
end

% caffe mode setting
phase = 'test'; % run with phase test (so that dropout isn't applied)

% --- settings end here ---

% outputs
result_folder = './analysis/cifar10/a';
feat_test_file = sprintf('%s/feat-test.mat', result_folder);
feat_train_file = sprintf('%s/feat-train.mat', result_folder);
binary_test_file = sprintf('%s/binary-test.mat', result_folder);
binary_train_file = sprintf('%s/binary-train.mat', result_folder);

% map and precision outputs
map_file = sprintf('%s/map.txt', result_folder);
precision_file = sprintf('%s/precision-at-k.txt', result_folder);



% fc7 feat
LSHparam_fc7.nbits = 36;
LSHparam_fc7.dim = 4096;
LSHparam_fc7 = trainLSH(LSHparam_fc7);

% input feat
LSHparam_ipt.nbits = 24;
LSHparam_ipt.dim = 32*32*3;
LSHparam_ipt = trainLSH(LSHparam_ipt);

% ae feat
aec_file = sprintf('%s/autoenc.mat', result_folder);
AEHparam_fc7.nbits = 36;
AEHparam_fc7.dim = 4096;
if exist('autoenc', 'var')
    AEHparam_fc7.aec = autoenc;
    clear autoenc
elseif exist(aec_file, 'file')
    load(aec_file,'autoenc');
    AEHparam_fc7.aec = autoenc;
    clear autoenc
end
% AEHparam_fc7 = trainAEH(AEHparam_fc7);

% ae input
aec_file = sprintf('%s/autoenc.mat', result_folder);
AEHparam_ipt.nbits = 24;
AEHparam_ipt.dim = 32*32*3;
if exist('autoenc', 'var')
    AEHparam_ipt.aec = autoenc;
    clear autoenc
elseif exist(aec_file, 'file')
    load(aec_file,'autoenc');
    AEHparam_ipt.aec = autoenc;
    clear autoenc
end
% AEHparam_ipt = trainAEH(AEHparam_ipt);


% param.pos = [1:1:40 50:2:100]; % The number of retrieved samples: Recall-The number of retrieved samples curve


% feature extraction- test set
use_saved = 1;
if exist(binary_test_file, 'file') ~= 0 && use_saved
    load(binary_test_file);
else
    if ~exist('ft_fc7_tst', 'var') || ~exist('ipt_tst', 'var')
        [feat_test, ft_fc7_tst, ipt_tst] = feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
        save ./analysis/a/ipt_tst ipt_tst
        save ./analysis/a/ft_fc7_tst ft_fc7_tst
    end
%     b_fc7_tst = compressLSH(ft_fc7_tst', LSHparam_fc7);
    %  b_fc7_tst = compressAEH(ft_fc7_tst', AEHparam_fc7);
%    b_ipt_tst = compressAEH(ipt_tst', AEHparam_ipt);
%     b_ipt_tst = compressLSH(ipt_tst', LSHparam_ipt);
    save(feat_test_file, 'feat_test', '-v7.3');
    binary_test = (feat_test>0.5);
%     binary_test = [binary_test; b_fc7_tst'; b_ipt_tst'];
%    binary_test = [binary_test; b_ipt_tst'];
    %  binary_test = [binary_test; b_fc7_tst'];
    binary_test = binary_test>0;
    save(binary_test_file,'binary_test','-v7.3');
end

% feature extraction- training set
if exist(binary_train_file, 'file') ~= 0 && use_saved
    load(binary_train_file);
else
    if ~exist('ft_fc7_trn', 'var') || ~exist('ipt_trn', 'var')
        [feat_train, ft_fc7_trn, ipt_trn]= feat_batch(use_gpu, model_def_file, model_file, train_file_list, feat_len);
        save ./analysis/a/ipt_trn ipt_trn
        save ./analysis/a/ft_fc7_trn ft_fc7_trn
    end
%     b_fc7_trn = compressLSH(ft_fc7_trn', LSHparam_fc7);
%      b_fc7_trn = compressAEH(ft_fc7_trn', AEHparam_fc7);
%    b_ipt_trn = compressAEH(ipt_trn', AEHparam_ipt);
%     b_ipt_trn = compressLSH(ipt_trn', LSHparam_ipt);
    save(feat_train_file, 'feat_train', '-v7.3');
    binary_train = (feat_train>0.5);
%     binary_train = [binary_train; b_fc7_trn'; b_ipt_trn'];
%    binary_train = [binary_train; b_ipt_trn'];
%      binary_train = [binary_train; b_fc7_trn'];
    binary_train = binary_train>0;
    save(binary_train_file,'binary_train','-v7.3');
end

%------------ train autoencoder ----------%
if 0   
    ae_trn_data = ft_fc7_trn;
    %ae_trn_data = ipt_trn;
%     ipt_trn = mapminmax(ipt_trn);
    ae_trn_data_batch = ae_trn_data(:,1:20000);
    autoenc = trainAutoencoder(ae_trn_data_batch,36,'useGPU',use_gpu,'SparsityProportion',0);
    net = network(autoenc);
    ae_trn_data_batch = ae_trn_data(:,20001:40000);
    net = train(net, ae_trn_data_batch, ae_trn_data_batch,'useGPU','yes');
    ae_trn_data_batch = ae_trn_data(:,40001:end);
    net = train(net, ae_trn_data_batch, ae_trn_data_batch,'useGPU','yes');
    clear ae_trn_data_batch ae_trn_data
    save ./analysis/a/autoenc.mat autoenc
end

trn_label = load(train_label_file);
tst_label = load(test_label_file);

visualize = 0;

if visualize
    param.numRetrieval = 48;

    param.query_id = 8840; %strange??
    param.query_id = 8866; %yello or red car?
    param.query_id = 8878;%8878;622 %good res
%     param.query_id = 8915; %bird8912
%     param.query_id = 2;24;45;64;88;94;110; the same dog
%     param.query_id = 3;%79;70;37;34;8;3 frog
    param.query_id = 631;%127;78;65;59;58;40; hourse
    

    similarity_file = sprintf('%s/sim_ind.mat', result_folder);
    
    if exist('sim_ind', 'var')
        
    elseif exist(similarity_file, 'file') ~= 0
        load(similarity_file)
    else
%         fprintf('computing hamming distance');
        similarity = hammingDist(binary_test',binary_train');
        save ./analysis/a/similarity  similarity
        [~,sim_ind]=sort(similarity,2);
        sim_ind = sim_ind(:, 1:param.numRetrieval);
        save ./analysis/a/sim_ind sim_ind
    end
    
    

    
    if (cifar10)
        retrieval_info = visualize_cifar10(1, sim_ind, param);
    else
        retrieval_info=visualize_bone(1, sim_ind, param);
    end
else
    % [map, precision_at_k] = precision(trn_label, binary_train, tst_label, binary_test, top_k,1,1);

    [map, precision_at_k] = precision(sim_ind, trn_label, tst_label, top_k);
    fprintf('MAP = %f\n',map);
    save(map_file, 'map', '-ascii');
    P = [[1:1:top_k]' precision_at_k'];
    save(precision_file, 'P', '-ascii');
end


