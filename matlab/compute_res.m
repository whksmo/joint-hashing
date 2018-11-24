function [mAP, rec, pre, retrieved_list] = compute_res(exp_data, param, method_name)
% input: 
%          data: 
%              data.train_data
%              data.test_data
%          param:
%              param.nbits---encoding length
%              param.pos---position
%          method: encoding length
% output:
%            recall: recall rate
%            precision: precision rate
%            evaluation_info: 
tic
train_data = exp_data.train_data;
test_data = exp_data.test_data;
trueRank = exp_data.knn_p2;
db_name = exp_data.db_name;

% compute Hamming metric and compute recall precision
root_path = './analysis/';
read_path = [root_path, db_name, '/', method_name, '/'];
% fprintf('%s',read_path);

switch(param.choice)
case 'evaluation'
    ind_file_name = 'sim_ind_full';
case 'visualization'
    ind_file_name = 'sim_ind';
end
sim_file_name = 'similarity';
if param.use_tree
    sim_file_name = [sim_file_name, '_tree'];
    ind_file_name = [ind_file_name, '_tree'];
end
sim_file_name = [sim_file_name, '.mat'];
ind_file_name = [ind_file_name, '.mat'];


switch(param.choice)
case 'evaluation'
    fprintf('start evaluating performance of %s\n', method_name);
    if exist([read_path, ind_file_name],'file') && param.use_saved_sim
        fprintf('use exist %s\n', ind_file_name);
        load([read_path, ind_file_name], 'sim_ind_full');
    elseif exist([read_path, sim_file_name], 'file') && param.use_saved_sim
        fprintf('generate %s from similarity matrix\n', ind_file_name);
        load([read_path, sim_file_name], 'similarity');
        [~, sim_ind_full] = sort(similarity, 2);
        sim_ind_full = sim_ind_full(:, 1:0.02*size(train_data,1));
        sim_ind_full = uint32(sim_ind_full);
        save([read_path, ind_file_name], 'sim_ind_full','-v7.3');
    else
        fprintf('unable to local the hamming matrix for method %s, try to create it.\n', method_name);
        [~, sim_ind_full] = create_sim_ind(exp_data, method_name, param, read_path);
        sim_ind_full = sim_ind_full(:, 1:0.02*size(train_data,1));
        sim_ind_full = uint32(sim_ind_full);
        save([read_path, ind_file_name], 'sim_ind_full','-v7.3');
    end
    
    clear train_data test_data;
    rec = [];
    pre = [];
    mAP = [];
    retrieved_list = [];
    
    if param.eval_map
        eva_info = eva_ranking(param, sim_ind_full, exp_data.train_label, exp_data.test_label);
        mAP = eva_info.mAP;
    end
    if param.eval_pr
        eva_info = eva_ranking(param, sim_ind_full, trueRank);
        rec = eva_info.recall;
        pre = eva_info.precision;
    end
case 'visualization'
    fprintf('Visualizing the results of %s\n', method_name);
        if exist([read_path, ind_file_name],'file')
            fprintf('use exist %s\n', ind_file_name);
            load([read_path, ind_file_name],'ind');
            sim_ind = ind;
            toc
        elseif exist([read_path, sim_file_name], 'file')
            fprintf('generate %s by similarity matrix\n', ind_file_name);
            load([read_path, sim_file_name], 'similarity');
            [~, sim_ind] = sort(similarity, 2);
            sim_ind = sim_ind(:, 1:param.numRetrieval);
            sim_ind = uint32(sim_ind);
            save([read_path, ind_file_name], 'sim_ind');
            toc
        else
            fprintf('unable to local the hamming matrix for method %s, try to create it.\n', method_name);
            [~, sim_ind] = create_sim_ind(exp_data, method_name, param, read_path);
            sim_ind = sim_ind(:, 1:param.numRetrieval);
            sim_ind = uint32(sim_ind);
            save([read_path, ind_file_name], 'sim_ind');
        end
        show_truth = false;
        retrieved_list = visualize_cifar10(sim_ind, param);
        if show_truth
            retrieved_true_list = visualize_cifar10(trueRank, param);
        end
        rec = [];
        pre = [];
        mAP = [];
end
toc
% fprintf('finishing evaluation for %s\n', method_name);
end