function [map precision_at_k] = precision(varargin)   
narginchk(2,7);
if nargin == 4
    rank = varargin{1};
    trn_label = varargin{2};
    tst_label = varargin{3};
    top_k = varargin{4};
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
    map = mean(AP);

elseif nargin < 8
    trn_label = varargin{1};
    trn_binary = varargin{2};
    tst_label = varargin{3};
    tst_binary = varargin{4};
    top_k = varargin{5};
    mode = varargin{6};
    print = varargin{7};

    num_tst = size(tst_binary,2);
    correct = zeros(top_k,1);
    total = zeros(top_k,1);
    error = zeros(top_k,1);
    AP = zeros(num_tst,1);

    Ns = 1:1:top_k;
    sum_tp = zeros(1, top_k);
    similarity = zeros(numel(trn_label),numel(tst_label));

    trn_binary=double(trn_binary);
    tst_binary=double(tst_binary);
    for i = 1:num_tst
        query_label = tst_label(i);
        if (print)
            fprintf('query %d\n',i);
        end
        query_binary = tst_binary(:,i);
        if mode==1
            if(print)
                tic
            end
            similarity = pdist2(trn_binary',query_binary','hamming');
            if(print)
                toc
                fprintf('Complete Query [Hamming] %.2f seconds\n',toc);
            end
        elseif mode ==2
        tic
        similarity = pdist2(trn_binary',query_binary','euclidean');
        toc
        if(print)
            fprintf('Complete Query [Euclidean] %.2f seconds\n',toc);
        end
        end
        % mAP: for i-th query, sort the retrieval resluts by hamming distance
        %       then, replace the retrieval resultw with a binary vector that represents 
        %       if k-th retrieval sample equals to the query.
        %       cumulatively sum the vector and divided it by current retrieval num to get precision
        %       AP are abtained by averaging precision
        %       mAP
        [x2,y2]=sort(similarity);
        
        buffer_yes = zeros(top_k,1);
        buffer_total = zeros(top_k,1);
        total_relevant = 0;
        
        for j = 1:top_k
            retrieval_label = trn_label(y2(j));
            
            if (query_label==retrieval_label)
                buffer_yes(j,1) = 1;
                total_relevant = total_relevant + 1;
            end
            buffer_total(j,1) = 1;
        end
        
        % compute precision
        P = cumsum(buffer_yes) ./ Ns';
        
    if (sum(buffer_yes) == 0)
        AP(i) = 0;
    else
        AP(i) = sum(P.*buffer_yes) / sum(buffer_yes);
    end
    
    sum_tp = sum_tp + cumsum(buffer_yes)';
        
    end

    precision_at_k = sum_tp ./ (Ns * num_tst);
    map = mean(AP);
end
end