%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the main script ufor evaluate the performance, and you can
% get Precision-Recall curve, mean Average Precision (mAP) curves, 
% Recall-The number of retrieved samples curve.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.choice = 'evaluation';
% param.choice = 'visualization';
param.numRetrieval=48;
% param.query_id = 595;%889;40% 816;1018  1025
param.query_id = 199;%15;
param.top_k = 587;
param.use_gpu = true;
param.use_saved = true;
param.eval_map = true;
param.eval_pr = 1;
param.use_saved_sim = true;
param.choose_label = 1;
param.pos = [1:10:40 50:100:1000 1000]; % The number of retrieved samples: Recall-The number of retrieved samples curve

db_info.name = 'cifar100';
% db_info.type = 'hog';
db_info.type = 'euclidean';
% db_info.type = 'ssim';
hashmethods = {'SSDH36+LSH12','SSDH24+LSH24','SSDH48'};

% different unsupervised methods with HoG feature
% hashmethods = { 'SSDH24+LSH24',  'SSDH24+ORI24', 'SSDH48'}; 
% db_info.type = 'hog';

% different unsupervised methods with Euclidean feature
% hashmethods = {'SSDH24+AEH24',  'SSDH24+ORI24', 'SSDH48'}; 
% db_info.type = 'euclidean';


% different ratio
% hashmethods = {'SSDH12+LSH36','SSDH24+LSH24', 'SSDH32+LSH16', 'SSDH48'}; 
% db_info.type = 'hog';

nhmethods = length(hashmethods);
tree_like = zeros(1, nhmethods);
% using tree-like searching strategy
% hashmethods = {'SSDH24+LSH24','SSDH24+LSH24', 'SSDH24+HOG24','SSDH24+HOG24', 'SSDH32+LSH16', 'SSDH32+LSH16', 'SSDH48'}; 
% db_info.type = 'hog';
% nhmethods = length(hashmethods);
% tree_like = zeros(1, nhmethods);
% tree_like(2) = 1;
% tree_like(4) = 1;
% tree_like(6) = 1;


% hashmethods = {'MODIFY24+LSH24'};
% hashmethods = {'SSDH12+AEH36'};
% hashmethods = {'SSDH_cifar10'};
% hashmethods = {'SSDH32+LSH16'};
% hashmethods = {'MODIFY32+LSH16'};


% loopnbits = [16 32 64 128];
loopnbits = [48];
runtimes = 1;





for k = 1:runtimes
    fprintf('The %d run time\n', k);

    if ~exist('exp_data','var') || ...
        ~strcmp(exp_data.db_name, db_info.name) || ...
        ~strcmp(exp_data.ft_type, db_info.type)
        exp_data = construct_data(db_info);
    end
    for i =1:length(loopnbits)
        fprintf('======start %d bits encoding======\n\n', loopnbits(i));
        param.nbits = loopnbits(i);
        for j = 1:nhmethods
            param.use_tree = tree_like(j);
            [mAP{k}{i,j}, rec{k}{i, j}, pre{k}{i, j}, retrieval_list{1,j}] = ...
                        compute_res(exp_data, param, hashmethods{1, j});
            if strcmp(param.choice, 'evaluation')
                fprintf('map for %s is: %f\n', hashmethods{1,j}, mAP{k}{i,j});
            end
        end
    end
end



% average MAP
% for j = 1:nhmethods
%     for i =1: length(loopnbits)
%         tmp = zeros(size(mAP{1, 1}{i, j}));
%         for k =1:runtimes
%             tmp = tmp+mAP{1, k}{i, j};
%         end
%         MAP{i, j} = tmp/runtimes;
%     end
%     clear tmp;
% end

% show precision vs. the number of retrieved sample.
if param.eval_pr
    if strcmp(param.choice,'evaluation')
        figure('Color', [1 1 1]);hold on;

        % plot attribution
        line_width=1;
        marker_size=6;
        xy_font_size=12;
        legend_font_size=12;
        linewidth = 1;
        title_font_size=xy_font_size;
        %
        choose_bits = 1; % i: choose the bits to show
        choose_times = 1; % k is the times of run times

        for j = 1: nhmethods
            pos = param.pos;
            prec = pre{choose_times}{choose_bits, j};
            %p = plot(pos(1,1:posEnd), recc(1,1:posEnd));
            p = plot(pos(1,1:end), prec(1,1:end));
            color = gen_color(j);
            marker = gen_marker(j);
            set(p,'Color', color)
            set(p,'Marker', marker);
            set(p,'LineWidth', line_width);
            set(p,'MarkerSize', marker_size);
        end
        
        str_nbits =  num2str(loopnbits(choose_bits));
        set(gca, 'linewidth', linewidth);
        h1 = xlabel('The number of retrieved samples');
        h2 = ylabel(['Precision @ ', str_nbits, ' bits']);
        title(db_info.name, 'FontSize', title_font_size);
        set(h1, 'FontSize', xy_font_size);
        set(h2, 'FontSize', xy_font_size);
        hleg = legend(hashmethods);
        set(hleg, 'FontSize', legend_font_size);
        set(hleg,'Location', 'best');
        axis square;
        box on;
        grid on;
        hold off;
    end
    
end

return





%% Try by myself
global aec
autoenc = trainAutoencoder(exp_data.db_data',20*param.nbits,...
        'EncoderTransferFunction','satlin',...
        'DecoderTransferFunction','purelin',...
        'L2WeightRegularization',0.004,...
        'SparsityRegularization',4,...
        'SparsityProportion',0.10);
aec=autoenc;

%%

hashmethods = {'FSH'};
%hashmethods = {'Our Method', 'SELVE', 'CBE-opt', 'LSH', 'PCAH', 'SH', 'SKLSH', 'DSH', 'SpH'};
%hashmethods = {'CBE-rand', 'CBE-opt', 'ITQ', 'LSH', 'PCAH', 'SH', 'SKLSH', 'PCA-RR', 'DSH', 'SpH'};
[~, ~, ~, ~,~, retrieval_list]=demo(exp_data, param, 'LSH');
%%
retrieval_list =  visualization(Dhamm, ID,47, exp_data,1);
retrieval_data1=imgdataRGB(retrieval_list(:,1),:,:);
retrieval_data2=imgdataRGB(retrieval_list(:,2),:,:);
fig('width',30)
subplot(1,2,1);
display_network(retrieval_data1);
subplot(1,2,2);

display_network(retrieval_data2);
%% show recall vs. the number of retrieved sample.
figure('Color', [1 1 1]); hold on;
posEnd = 8;
choose_bits=1;
for j = 1: nhmethods
    pos = param.pos;
    recc = rec{choose_times}{choose_bits, j};
    %p = plot(pos(1,1:posEnd), recc(1,1:posEnd));
    p = plot(pos(1,1:end), recc(1,1:end));
    color = gen_color(j);
    marker = gen_marker(j);
    set(p,'Color', color)
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

str_nbits =  num2str(loopnbits(choose_bits));
set(gca, 'linewidth', linewidth);
h1 = xlabel('Number of top returned images');
% h2 = ylabel(['Recall  ', str_nbits, ' bits']);
h2 = ylabel('Recall');
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg,'Location', 'best');
box on;
grid on;
hold off;

%% show precision vs. the number of retrieved sample.
choose_bits=4;
fig('Color', [1 1 1]);
hold on;
posEnd = 8;
for j = 1: nhmethods
    pos = param.pos;
    prec = pre{choose_times}{choose_bits, j};
    %p = plot(pos(1,1:posEnd), recc(1,1:posEnd));
    p = plot(pos(1,1:end), prec(1,1:end));
    color = gen_color(j);
    marker = gen_marker(j);
    set(p,'Color', color)
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

str_nbits =  num2str(loopnbits(choose_bits));
set(gca, 'linewidth', linewidth);
h1 = xlabel('Number of top returned images');
% h2 = ylabel(['Precision  ', str_nbits, ' bits']);
h2 = ylabel('Precision');
% title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
% axis([0 1000 0.15 0.9]);
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg,'Location', 'best');
box on;
grid on;
hold off;

%% show precision vs. recall , i is the selection of which bits.

% figure('Color', [1 1 1]); 
fig('width',15)
hold on;
choose_bits=1;
for j = 1: nhmethods
    p = plot(rec{choose_times}{choose_bits, j}, pre{choose_times}{choose_bits, j});
    color=gen_color(j);
    marker=gen_marker(j);
    set(p,'Color', color)
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

str_nbits =  num2str(loopnbits(choose_bits));
h1 = xlabel(['Recall  ', str_nbits, ' bits']);
h2 = ylabel('Precision');
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg,'Location', 'best');
set(gca, 'linewidth', linewidth);
box on;
grid on;
hold off;
%% show multi precision vs. runtime is the selection of which bits.
fig('Color', [1 1 1]); hold on;
choose_bits=1;
ind=6;
for j=1:nhmethods
    for i=1:runtimes
        diff_time(i,j)=pre{i}{choose_bits, j}(ind);
    end
end
for j = 1: nhmethods
    p = plot((1:runtimes),diff_time(:,j));
    color=gen_color(j);
    marker=gen_marker(j);
    set(p,'Color', color)
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

str_nbits =  num2str(loopnbits(choose_bits));
h1 = xlabel(['Runtimes']);
h2 = ylabel(['Precision @ ' num2str(pos(ind)) 'retrieval']);
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg,'Location', 'best');
set(gca, 'linewidth', linewidth);
box on;
grid on;
hold off;
clear ind diff_time
%% show mAP. This mAP function is provided by Yunchao Gong
figure('Color', [1 1 1]); hold on;
for j = 1: nhmethods
    map = [];
    for i = 1: length(loopnbits)
        map = [map, MAP{i, j}];
    end
    p = plot(log2(loopnbits), map);
    color=gen_color(j);
    marker=gen_marker(j);
    set(p,'Color', color);
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

h1 = xlabel('Number of bits');
h2 = ylabel('mean Average Precision (mAP)');
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
set(gca, 'xtick', log2(loopnbits));
set(gca, 'XtickLabel', {'8', '16', '32', '64', '128'});
set(gca, 'linewidth', linewidth);
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg, 'Location', 'best');
box on;
grid on;
hold off;