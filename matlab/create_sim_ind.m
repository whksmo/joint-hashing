function [similarity, sim_ind_full] = create_sim_ind(exp_data, method_name, param, result_folder)

tic
train_data = exp_data.train_data;
test_data = exp_data.test_data;
db_name = exp_data.db_name;
if nargin == 3
    root_path = './analysis/';
    result_folder = [root_path, db_name, '/', method_name, '/'];
end
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end
feat_test_file = sprintf('%s/feat-test.mat', result_folder);
feat_train_file = sprintf('%s/feat-train.mat', result_folder);
binary_test_file = sprintf('%s/binary-test.mat', result_folder);
binary_train_file = sprintf('%s/binary-train.mat', result_folder);
map_file = sprintf('%s/map.txt', result_folder);
precision_file = sprintf('%s/precision-at-k.txt', result_folder);
use_saved = param.use_saved;

if strfind(db_name, 'cifar100')
    test_file_list = './data/cifar100/test-file-list.txt';
    test_label_file = './data/cifar100/test-label.txt';
    train_file_list = './data/cifar100/train-file-list.txt';
    train_label_file = './data/cifar100/train-label.txt';
elseif strfind(db_name, 'cifar10')
    test_file_list = './data/cifar10/test-file-list.txt';
    test_label_file = './data/cifar10/test-label.txt';
    train_file_list = './data/cifar10/train-file-list.txt';
    train_label_file = './data/cifar10/train-label.txt';
elseif strfind(db_name, 'bone')
    test_file_list = './data/bone/test-file-list.txt';
    train_file_list = './data/bone/train-file-list.txt';
    test_label_file = './data/bone/test-label-bone.txt';
    train_label_file = './data/bone/train-label-bone.txt';
else
    fprintf('error, there is no such data sets\n');
end
% trn_label = load(train_label_file);
% tst_label = load(test_label_file);

phase = 'test';
fprintf('......%s start...... \n\n', method_name);
use_gpu = param.use_gpu;
switch(method_name)

case 'MODIFY24+LSH24'
    feat_len = 24;
    model_file = sprintf('./examples/cifar10/cifar10_%d_iter_20000.caffemodel', feat_len);
    model_def_file = sprintf('./examples/bone-finetune/deploy_%d.prototxt', feat_len);
    
    LSHparam_fc7.nbits = param.nbits - feat_len;
    LSHparam_fc7.dim = 4096;
    LSHparam_fc7 = trainLSH(LSHparam_fc7);
    
    if exist(binary_test_file, 'file') ~= 0 && use_saved
        load(binary_test_file);
    else
        [feat_test, ft_fc7_tst, ipt_tst] = feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
        save([result_folder, 'ipt_tst'], 'ipt_tst');
        save([result_folder, 'ft_fc7_tst'], 'ft_fc7_tst');
        b_fc7_tst = compressLSH(ft_fc7_tst', LSHparam_fc7);
        % b_fc7_tst = compressAEH(ft_fc7_tst', AEHparam_fc7);
        % b_ipt_tst = compressAEH(ipt_tst', AEHparam_ipt);
        % b_ipt_tst = compressLSH(ipt_tst', LSHparam_ipt);
        save(feat_test_file, 'feat_test', '-v7.3');
        binary_test = (feat_test>0.5);

        % binary_test = [binary_test; b_fc7_tst'; b_ipt_tst'];
        % binary_test = [binary_test; b_ipt_tst'];
        binary_test = [binary_test; b_fc7_tst'];
        binary_test = binary_test>0;
        save(binary_test_file,'binary_test','-v7.3');
    end
    
    if exist(binary_train_file, 'file') ~= 0 && use_saved
        load(binary_train_file);
    else
        [feat_train, ft_fc7_trn, ipt_trn]= feat_batch(use_gpu, model_def_file, model_file, train_file_list, feat_len);
        save([result_folder, 'ipt_trn'], 'ipt_trn');
        save([result_folder, 'ft_fc7_trn'], 'ft_fc7_trn');
        b_fc7_trn = compressLSH(ft_fc7_trn', LSHparam_fc7);
        % b_fc7_trn = compressAEH(ft_fc7_trn', AEHparam_fc7);
        % b_ipt_trn = compressAEH(ipt_trn', AEHparam_ipt);
        % b_ipt_trn = compressLSH(ipt_trn', LSHparam_ipt);
        save(feat_train_file, 'feat_train', '-v7.3');
        binary_train = (feat_train>0.5);
        % binary_train = [binary_train; b_fc7_trn'; b_ipt_trn'];
        % binary_train = [binary_train; b_ipt_trn'];
        binary_train = [binary_train; b_fc7_trn'];
        binary_train = binary_train>0;
        save(binary_train_file,'binary_train','-v7.3');
    end
case 'MODIFY32+LSH16'
    feat_len = 32;
    model_file = sprintf('./examples/cifar10/cifar10_%d_iter_20000.caffemodel', feat_len);
    model_def_file = sprintf('./examples/bone-finetune/deploy_%d.prototxt', feat_len);
    
    LSHparam_fc7.nbits = param.nbits - feat_len;
    LSHparam_fc7.dim = 4096;
    LSHparam_fc7 = trainLSH(LSHparam_fc7);
    
    if exist(binary_test_file, 'file') ~= 0 && use_saved
        load(binary_test_file);
    else
        [feat_test, ft_fc7_tst, ipt_tst] = feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
        save([result_folder, 'ft_fc7_tst'], 'ft_fc7_tst');
        b_fc7_tst = compressLSH(ft_fc7_tst', LSHparam_fc7);
        save(feat_test_file, 'feat_test', '-v7.3');
        binary_test = (feat_test>0.5);
        binary_test = [binary_test; b_fc7_tst'];
        binary_test = binary_test>0;
        save(binary_test_file,'binary_test','-v7.3');
    end
    
    if exist(binary_train_file, 'file') ~= 0 && use_saved
        load(binary_train_file);
    else
        [feat_train, ft_fc7_trn, ipt_trn]= feat_batch(use_gpu, model_def_file, model_file, train_file_list, feat_len);
        save([result_folder, 'ft_fc7_trn'], 'ft_fc7_trn');
        b_fc7_trn = compressLSH(ft_fc7_trn', LSHparam_fc7);
        save(feat_train_file, 'feat_train', '-v7.3');
        binary_train = (feat_train>0.5);
        binary_train = [binary_train; b_fc7_trn'];
        binary_train = binary_train>0;
        save(binary_train_file,'binary_train','-v7.3');
    end

case 'SSDH12+LSH36'
    feat_len = 12;
    param.sp_bits = feat_len;
    model_file = sprintf('./examples/cifar10/cifar10_%d_iter_50000.caffemodel', feat_len);
    model_def_file = sprintf('./examples/bone-finetune/deploy_%d.prototxt', feat_len);
    
    LSHparam_fc7.nbits = param.nbits - feat_len;
    LSHparam_fc7.dim = 4096;
    LSHparam_fc7 = trainLSH(LSHparam_fc7);
    
    if exist(binary_test_file, 'file') ~= 0 && use_saved
        load(binary_test_file);
    else
        [feat_test, ft_fc7_tst, ipt_tst] = feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
        save([result_folder, 'ft_fc7_tst'], 'ft_fc7_tst');
        b_fc7_tst = compressLSH(ft_fc7_tst', LSHparam_fc7);
        save(feat_test_file, 'feat_test', '-v7.3');
        binary_test = (feat_test>0.5);
        binary_test = [binary_test; b_fc7_tst'];
        binary_test = binary_test>0;
        save(binary_test_file,'binary_test','-v7.3');
    end
    
    if exist(binary_train_file, 'file') ~= 0 && use_saved
        load(binary_train_file);
    else
        [feat_train, ft_fc7_trn, ipt_trn]= feat_batch(use_gpu, model_def_file, model_file, train_file_list, feat_len);
        save([result_folder, 'ft_fc7_trn'], 'ft_fc7_trn');
        b_fc7_trn = compressLSH(ft_fc7_trn', LSHparam_fc7);
        save(feat_train_file, 'feat_train', '-v7.3');
        binary_train = (feat_train>0.5);
        binary_train = [binary_train; b_fc7_trn'];
        binary_train = binary_train>0;
        save(binary_train_file,'binary_train','-v7.3');
    end
case 'SSDH24+LSH24'
    feat_len = 24;
    param.sp_bits = feat_len;
    model_file = sprintf('./examples/cifar10/cifar10_%d_iter_50000.caffemodel', feat_len);
    model_def_file = sprintf('./examples/bone-finetune/deploy_%d.prototxt', feat_len);
    
    LSHparam_fc7.nbits = param.nbits - feat_len;
    LSHparam_fc7.dim = 4096;
    LSHparam_fc7 = trainLSH(LSHparam_fc7);
    
    if exist(binary_test_file, 'file') ~= 0 && use_saved
        load(binary_test_file);
    else
        [feat_test, ft_fc7_tst, ipt_tst] = feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
        save([result_folder, 'ft_fc7_tst'], 'ft_fc7_tst');
        b_fc7_tst = compressLSH(ft_fc7_tst', LSHparam_fc7);
        save(feat_test_file, 'feat_test', '-v7.3');
        binary_test = (feat_test>0.5);
        binary_test = [binary_test; b_fc7_tst'];
        binary_test = binary_test>0;
        save(binary_test_file,'binary_test','-v7.3');
    end
    
    if exist(binary_train_file, 'file') ~= 0 && use_saved
        load(binary_train_file);
    else
        [feat_train, ft_fc7_trn, ipt_trn]= feat_batch(use_gpu, model_def_file, model_file, train_file_list, feat_len);
        save([result_folder, 'ft_fc7_trn'], 'ft_fc7_trn');
        b_fc7_trn = compressLSH(ft_fc7_trn', LSHparam_fc7);
        save(feat_train_file, 'feat_train', '-v7.3');
        binary_train = (feat_train>0.5);
        binary_train = [binary_train; b_fc7_trn'];
        binary_train = binary_train>0;
        save(binary_train_file,'binary_train','-v7.3');
    end
case 'SSDH32+LSH16'
    feat_len = 32;
    param.sp_bits = feat_len;
    model_file = sprintf('./examples/cifar10/cifar10_%d_iter_50000.caffemodel', feat_len);
    model_def_file = sprintf('./examples/bone-finetune/deploy_%d.prototxt', feat_len);
    
    LSHparam_fc7.nbits = param.nbits - feat_len;
    LSHparam_fc7.dim = 4096;
    LSHparam_fc7 = trainLSH(LSHparam_fc7);
    
    if exist(binary_test_file, 'file') ~= 0 && use_saved
        load(binary_test_file);
    else
        [feat_test, ft_fc7_tst, ipt_tst] = feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
        save([result_folder, 'ft_fc7_tst'], 'ft_fc7_tst');
        b_fc7_tst = compressLSH(ft_fc7_tst', LSHparam_fc7);
        save(feat_test_file, 'feat_test', '-v7.3');
        binary_test = (feat_test>0.5);
        binary_test = [binary_test; b_fc7_tst'];
        binary_test = binary_test>0;
        save(binary_test_file,'binary_test','-v7.3');
    end
    
    if exist(binary_train_file, 'file') ~= 0 && use_saved
        load(binary_train_file);
    else
        [feat_train, ft_fc7_trn, ipt_trn]= feat_batch(use_gpu, model_def_file, model_file, train_file_list, feat_len);
        save([result_folder, 'ft_fc7_trn'], 'ft_fc7_trn');
        b_fc7_trn = compressLSH(ft_fc7_trn', LSHparam_fc7);
        save(feat_train_file, 'feat_train', '-v7.3');
        binary_train = (feat_train>0.5);
        binary_train = [binary_train; b_fc7_trn'];
        binary_train = binary_train>0;
        save(binary_train_file,'binary_train','-v7.3');
    end
case 'SSDH24+ORI24'

    assert(strcmp(exp_data.ft_type, 'euclidean'));
    feat_len = 24;
    param.sp_bits = feat_len;
    model_file = sprintf('./examples/cifar10/cifar10_%d_iter_50000.caffemodel', feat_len);
    model_def_file = sprintf('./examples/bone-finetune/deploy_%d.prototxt', feat_len);
    
    LSHparam_ipt.nbits = param.nbits - feat_len;
    LSHparam_ipt.dim = 32 * 32 * 3;
    LSHparam_ipt = trainLSH(LSHparam_ipt);

    num_training = size(train_data, 1);
    XX = [train_data; test_data];
    sampleMean = mean(XX,1);
    XX = (double(XX)-repmat(sampleMean,size(XX,1),1));
    train_data = XX(1:num_training, :);
    test_data = XX(num_training+1:end, :);

    if exist(binary_test_file, 'file') ~= 0 && use_saved
        load(binary_test_file);
    else
        [feat_test, ft_fc7_tst, ipt_tst] = feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
        b_ipt_tst = compressLSH(test_data, LSHparam_ipt);
        save(feat_test_file, 'feat_test', '-v7.3');
        binary_test = (feat_test>0.5);
        binary_test = [binary_test; b_ipt_tst'];
        binary_test = binary_test>0;
        save(binary_test_file,'binary_test','-v7.3');
    end
    
    if exist(binary_train_file, 'file') ~= 0 && use_saved
        load(binary_train_file);
    else
        [feat_train, ft_fc7_trn, ipt_trn]= feat_batch(use_gpu, model_def_file, model_file, train_file_list, feat_len);
        b_ipt_trn = compressLSH(train_data, LSHparam_ipt);
        save(feat_train_file, 'feat_train', '-v7.3');
        binary_train = (feat_train>0.5);
        binary_train = [binary_train; b_ipt_trn'];
        binary_train = binary_train>0;
        save(binary_train_file,'binary_train','-v7.3');
    end
case 'SSDH24+HOG24'
    if ~strcmp(exp_data.ft_type, 'hog')
        hog_data = load(['./data/', db_name, '/cifar_hog.mat', ]);
        train_data = hog_data.train_data;
        test_data = hog_data.test_data;
    end
    feat_len = 24;
    param.sp_bits = feat_len;
    model_file = sprintf('./examples/cifar10/cifar10_%d_iter_50000.caffemodel', feat_len);
    model_def_file = sprintf('./examples/bone-finetune/deploy_%d.prototxt', feat_len);
    
    num_training = size(train_data, 1);
    XX = [train_data; test_data];
    sampleMean = mean(XX,1);
    XX = (double(XX)-repmat(sampleMean,size(XX,1),1));
    train_data = XX(1:num_training, :);
    test_data = XX(num_training+1:end, :);

    LSHparam_ipt.nbits = param.nbits - feat_len;
    LSHparam_ipt.dim = size(test_data, 2);
    LSHparam_ipt = trainLSH(LSHparam_ipt);
    
    if exist(binary_test_file, 'file') ~= 0 && use_saved
        load(binary_test_file);
    else
        feat_test = feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
        b_ipt_tst = compressLSH(test_data, LSHparam_ipt);
        save(feat_test_file, 'feat_test', '-v7.3');
        binary_test = (feat_test>0.5);
        binary_test = [binary_test; b_ipt_tst'];
        binary_test = binary_test>0;
        save(binary_test_file,'binary_test','-v7.3');
    end
    
    if exist(binary_train_file, 'file') ~= 0 && use_saved
        load(binary_train_file);
    else
        feat_train = feat_batch(use_gpu, model_def_file, model_file, train_file_list, feat_len);
        b_ipt_trn = compressLSH(train_data, LSHparam_ipt);
        save(feat_train_file, 'feat_train', '-v7.3');
        binary_train = (feat_train>0.5);
        binary_train = [binary_train; b_ipt_trn'];
        binary_train = binary_train>0;
        save(binary_train_file,'binary_train','-v7.3');
    end
case 'SSDH48'
    feat_len = 48;
    model_file = sprintf('./examples/cifar10/cifar10_%d_iter_50000.caffemodel', feat_len);
    model_def_file = sprintf('./examples/bone-finetune/deploy.prototxt', feat_len);
    
    if exist(binary_test_file, 'file') ~= 0 && use_saved
        load(binary_test_file);
    else
        feat_test = feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
        save(feat_test_file, 'feat_test', '-v7.3');
        binary_test = (feat_test>0.5);
        binary_test = [binary_test];
        save(binary_test_file,'binary_test','-v7.3');
    end
    
    if exist(binary_train_file, 'file') ~= 0 && use_saved
        load(binary_train_file);
    else
        feat_train= feat_batch(use_gpu, model_def_file, model_file, train_file_list, feat_len);
        save(feat_train_file, 'feat_train', '-v7.3');
        binary_train = (feat_train>0.5);
        binary_train = [binary_train];
        save(binary_train_file,'binary_train','-v7.3');
    end

end

fprintf('computing hamming distance for %s\n', method_name);
similarity = hammingDist(binary_test',binary_train', param);
[~,sim_ind_full]=sort(similarity,2);
