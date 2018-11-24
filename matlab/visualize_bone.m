function nn_info = visualize_bone(show, ind, param)   

% QueryTimes = size(tst_binary,2);

% correct = zeros(K,1);
% total = zeros(K,1);
% error = zeros(K,1);
% AP = zeros(QueryTimes,1);
% 
% Ns = 1:1:K;
% sum_tp = zeros(1, length(Ns));
% similarity = zeros(numel(trn_label),numel(tst_label));

% for i = 1:QueryTimes
%     
%     query_label = tst_label(i);
%     fprintf('query %d\n',i);
%     query_binary = tst_binary(:,i);
%     if mode==1
%     tic
%     similarity = pdist2(trn_binary',query_binary','hamming');
%     toc
%     fprintf('Complete Query [Hamming] %.2f seconds\n',toc);
%     elseif mode ==2
%     tic
%     similarity = pdist2(trn_binary',query_binary','euclidean');
%     toc
%     fprintf('Complete Query [Euclidean] %.2f seconds\n',toc);
%     end
% 
%     [x2,y2]=sort(similarity);
%     
%     buffer_yes = zeros(K,1);
%     buffer_total = zeros(K,1);
%     total_relevant = 0;
%     
%     for j = 1:K
%         retrieval_label = trn_label(y2(j));
%         
%         if (query_label==retrieval_label)
%             buffer_yes(j,1) = 1;
%             total_relevant = total_relevant + 1;
%         end
%         buffer_total(j,1) = 1;
%     end
%     
%     % compute precision
%     P = cumsum(buffer_yes) ./ Ns';
%     
%    if (sum(buffer_yes) == 0)
%        AP(i) = 0;
%    else
%        AP(i) = sum(P.*buffer_yes) / sum(buffer_yes);
%    end
%    
%    sum_tp = sum_tp + cumsum(buffer_yes)';
%     
% end
% 
% precision_at_k = sum_tp ./ (Ns * QueryTimes);
% map = mean(AP);


load ./data/bone/cnn_each.mat cnn_each
load ./data/bone/from.mat from
load ./data/bone/position.mat position
load ./data/bone/label.mat label
load ./data/bone/sel.mat sel_test sel_train
top_k = param.numRetrieval;
query_id=param.query_id;  % 140: kidney!=left rib; 

% retrieval_info=visualize_result(1,query_id,ind(query_id,1:top_k),param,cnn_each,from,position,label_list);


imgsize = 64;
imgdata=zeros(top_k+1, imgsize * imgsize);
dataset_path = './data/bone';
retrieval_index = ind(query_id,1:top_k);

query_trueID=sel_test(query_id);
retrieval_index=sel_train(retrieval_index);

nn_info(1).imname=cnn_each(query_trueID).from;
if nn_info(1).imname(end-4)=='p'
    nn_info(1).pathname='./HS_SICK_AD/';
else
    nn_info(1).pathname='./HS_NORM_AD/';
end
nn_info(1).from=from(query_trueID).name;
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
nn_info(1).label=label(query_trueID);
nn_info(1).xCenter=from(query_trueID).xCenter;
nn_info(1).yCenter=from(query_trueID).yCenter;

imnames=[dataset_path nn_info(1).pathname(2:end) nn_info(1).imname];
img=imread(imnames);
imgdata(1,:)=reshape(img,1,[]);
for i=2:top_k+1
    nn_info(i).imname=cnn_each(retrieval_index(i-1)).from;
    nn_info(i).from=from(retrieval_index(i-1)).name;    
    if nn_info(i).imname(end-4)=='p'
        nn_info(i).pathname='./HS_SICK_AD/';
    else
        nn_info(i).pathname='./HS_NORM_AD/';
    end
    switch position(retrieval_index(i-1))
        case 0
            nn_info(i).position='Vertebrate';
        case 1
            nn_info(i).position='Left Rib';
        case 2
            nn_info(i).position='Right Rib';
        case 3
            nn_info(i).position='Scapula';
        case 4
            nn_info(i).position='Kidney';  
    end
    nn_info(i).label=label(retrieval_index(i-1));
    nn_info(i).xCenter=from(retrieval_index(i-1)).xCenter;
    nn_info(i).yCenter=from(retrieval_index(i-1)).yCenter;
    
    imnames=[dataset_path nn_info(i).pathname(2:end) nn_info(i).imname];
    img=imread(imnames);
    imgdata(i,:)=reshape(img,1,[]);
end
if show
    display_network(imgdata(1:end,:),false,true);
end



end