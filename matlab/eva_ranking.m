function [ evaluation_info ] = eva_ranking(param, rank, varargin)
if nargin == 4
    % fprintf('computing mAP\n');
    trn_label = varargin{1};
    tst_label = varargin{2};
    if size(trn_label,2) > 1
        fprintf('choose No.%d label for eval\n',param.choose_label);
        trn_label = trn_label(:,param.choose_label);
        tst_label = tst_label(:,param.choose_label);
    end
    top_k = param.top_k;
    Ns = 1:1:top_k;
    rank = rank(:, 1:top_k);
    rank = trn_label(rank);
    num_tst = numel(tst_label);
    AP = zeros(num_tst, 1);
    sum_tp = zeros(1, top_k);
    for i = 1 : num_tst
        buffer_yes = rank(i, :) == tst_label(i);
        P = cumsum(buffer_yes)/Ns;
        num_correct = sum(buffer_yes);
        if num_correct == 0
            AP(i) = 0;
        else
            AP(i) = sum(P.*buffer_yes) / sum(buffer_yes);
        end
        sum_tp = sum_tp + cumsum(buffer_yes);
    end
    precision_at_k = sum_tp ./ (Ns * num_tst);
    mAP = mean(AP);
    evaluation_info.mAP = mAP;
    evaluation_info.precision_at_k = precision_at_k;
else
    % fprintf('computing precision & recall\n');
    
    trueRank = varargin{1};
    
    [ntest, ~] = size(rank);
    
    M_set = param.pos;
    
    for n = 1:ntest  
        rank(n,:) = ismember(rank(n,:), trueRank(n,:));    
    end

    truth_num=size(trueRank,2);

    for i_M=1:length(M_set)
        % retrieval top M sample from trainsets
        M=M_set(i_M);

        Ntrue=sum(rank(:,1:M),2);

        Pi=Ntrue/M;
        % precision over all queries
        P(i_M)=mean(Pi,1);

        Ri=Ntrue/truth_num;
        R(i_M)=mean(Ri,1);
    end

    evaluation_info.recall=R;
    evaluation_info.precision=P;
end