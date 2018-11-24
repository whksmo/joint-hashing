%%
param.choice = 'evaluationO';
param.numRetrieval=6;
param.pos = [1:2:40 50:5:100]; % The number of retrieved samples: Recall-The number of retrieved samples curve
ID.train = exp_data.train_ID;
ID.test = exp_data.test_ID;
ID.query = 33;
%%
ID.query = [33];
ID.train = exp_data.train_ID;
ID.test = exp_data.test_ID;
retrieval_list =  visualization(Dhamm, ID,47, exp_data,1);
retrieval_data1=imgdataRGB(retrieval_list(:,1),:,:);

% subplot(1,2,1);
display_network(retrieval_data1);
% subplot(1,2,2);
% retrieval_data2=imgdataRGB(retrieval_list(:,2),:,:);
% display_network(retrieval_data2);


%%
numRetrieval=size(exp_data.knn_p2,2);
train_ID = ID.train;
test_ID = ID.test;
M_set = param.pos;
precision=zeros(1,length(M_set));
recall=zeros(1,length(M_set));
n=1;
for id=1:1
    query_ID = 21;
    queryDhamm = Dhamm(query_ID, :);
    ID_Dhamm= [train_ID; queryDhamm]';
    [ID_rankDhamm, index]= sortrows(ID_Dhamm, 2);
    query_trueID = test_ID(query_ID);
    %true_candidate:[numRetrieval*2],the first col is the index of train_data,second col is
    %the hamming distance to the query data
    true_candidate = ID_rankDhamm(1:numRetrieval, :);
    retrieval_index=true_candidate(:,1);
    % true_label=imglabels(query_trueID);
    % retrieval_labels=imglabels(retrieval_index);

%     true_label=label(query_trueID);
%     retrieval_labels=label(retrieval_index);
    true_label=position(query_trueID);
    retrieval_labels=position(retrieval_index);
    
    is_same=retrieval_labels==true_label;
     for i_M=1:length(M_set)
         M=M_set(i_M);

         Ntrue=sum(is_same(1:M));

         Pi=Ntrue/M;
         precision(i_M)=precision(i_M)+mean(Pi,1);

         Ri=Ntrue/numRetrieval;
         recall(i_M)=recall(i_M)+mean(Ri,1);
     end
end
precision=precision/n;
recall=recall/n;

%
%���ҵķ�ʽ�����ӻ�
num_visual=35;
imgData=zeros(num_visual+1,64*64);

nn_info(1).imname=cnn_each(query_trueID).from;
if nn_info(1).imname(end-4)=='p'
    nn_info(1).pathname='D:/projects/matlab project/bone_scan_code/ͼ�����/Retrieval/BS_Retrieval/HS_SICK_AD/';
else
    nn_info(1).pathname='D:/projects/matlab project/bone_scan_code/ͼ�����/Retrieval/BS_Retrieval/HS_NORM_AD/';
end
nn_info(1).from=from(query_trueID).name;
nn_info(1).xCenter=from(query_trueID).xCenter;
nn_info(1).yCenter=from(query_trueID).yCenter;
switch position(query_trueID)
    case 0
        nn_info(1).position='Vertebrate';
    case 1
        nn_info(1).position='Left Rib';
    case 2
        nn_info(1).position='Right Rib';
    case 3
        nn_info(1).position='Scapula';
    case 4
        nn_info(1).position='Kidney';  
end
imnames=[nn_info(1).pathname nn_info(1).imname];
img=imread(imnames);
imgData(1,:)=reshape(img,1,[]);
for jj=2:num_visual+1
    nn_info(jj).imname=cnn_each(retrieval_index(jj-1)).from;
    nn_info(jj).from=from(retrieval_index(jj-1)).name;
    nn_info(jj).xCenter=from(retrieval_index(jj)).xCenter;
    nn_info(jj).yCenter=from(retrieval_index(jj)).yCenter;
    if nn_info(jj).imname(end-4)=='p'
        nn_info(jj).pathname='D:/projects/matlab project/bone_scan_code/ͼ�����/Retrieval/BS_Retrieval/HS_SICK_AD/';
    else
        nn_info(jj).pathname='D:/projects/matlab project/bone_scan_code/ͼ�����/Retrieval/BS_Retrieval/HS_NORM_AD/';
    end
    switch position(retrieval_index(jj-1))
        case 0
            nn_info(jj).position='Vertebrate';
        case 1
            nn_info(jj).position='Left Rib';
        case 2
            nn_info(jj).position='Right Rib';
        case 3
            nn_info(jj).position='Scapula';
        case 4
            nn_info(jj).position='Kidney';  
    end
    imnames=[nn_info(jj).pathname nn_info(jj).imname];
    img=imread(imnames);
    imgData(jj,:)=reshape(img,1,[]);
end

display_network(imgData,false,true);

%% show precision vs. the number of retrieved sample.
figure('Color', [1 1 1]); hold on;
posEnd = 8;
for j = 1: 1
    pos = param.pos;
    p = plot(pos(1,1:end), precision(1,1:end));
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
%% show recall vs. the number of retrieved sample.
figure('Color', [1 1 1]); hold on;
posEnd = 8;
for j = 1: 1
    pos = param.pos;
    p = plot(pos(1,1:end), recall(1,1:end));
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
h2 = ylabel(['Recall @ ', str_nbits, ' bits']);
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

%%
num_visual=10;
for jj=1:num_visual
    nn_info(jj).imname=cnn_each(retrieval_index(jj)).from;
    if nn_info(jj).imname(end-4)=='p'
        nn_info(jj).pathname='D:/projects/matlab project/bone_scan_code/ͼ�����/Retrieval/BS_Retrieval/HS_SICK/';
    else
        nn_info(jj).pathname='D:/projects/matlab project/bone_scan_code/ͼ�����/Retrieval/BS_Retrieval/HS_NORM/';
    end
end
nn_info(num_visual+1).imname=cnn_each(query_trueID).from;
if nn_info(num_visual+1).imname(end-4)=='p'
    nn_info(num_visual+1).pathname='D:/projects/matlab project/bone_scan_code/ͼ�����/Retrieval/BS_Retrieval/HS_SICK/';
else
    nn_info(num_visual+1).pathname='D:/projects/matlab project/bone_scan_code/ͼ�����/Retrieval/BS_Retrieval/HS_NORM/';
end

title('Query Image');
imnames=[nn_info(num_visual+1).pathname nn_info(num_visual+1).imname];
img=imread(imnames);
subplot(1,num_visual+1,1);
imshow(img,[]);


for jj=1:num_visual
    imnames=[nn_info(jj).pathname nn_info(jj).imname];
    img=imread(imnames);
    subplot(1,num_visual+1,jj+1);
    imshow(img,[]);
%     title(['No.' num2str(jj) ' Result Image']);
end

%% plot attribution
line_width=2;
marker_size=8;
xy_font_size=14;
legend_font_size=12;
linewidth = 1.6;
title_font_size=xy_font_size;
loopnbits=64;
%choose_bits = 5; % i: choose the bits to show
%choose_times = 3; % k is the times of run times
choose_bits = 1; % i: choose the bits to show
choose_times = 1; % k is the times of run times



