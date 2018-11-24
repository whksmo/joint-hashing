function retrieved_list = visualization(Dhamm, ID, numRetrieval, exp_data,ground_truth)

% train_ID = ID.train;
% test_ID = ID.test;
query_ID = ID.query;
clear ID;
if (ground_truth)
    queryDhamm = Dhamm(query_ID, :);
    ID_Dhamm= [train_ID; queryDhamm]';
    [ID_rankDhamm, index]= sortrows(ID_Dhamm, 2);
    query_trueID = test_ID(query_ID);
    %true_candidate:[numRetrieval*2],the first col is the index of train_data,second col is
    %the hamming distance to the query data
    true_candidate = ID_rankDhamm(1:numRetrieval, :);
    
    %compute the distance matrix between the query data and the k-nearest
    %train data    [1*numRetrieval]
%     disQueryTraining = distMat(test_data(query_ID, :), train_data(index(1:numRetrieval, :), :));
    
    %the first col is the index of train data which are as the retrieval
    %results, the second col is the L2 distance to the query data
%     list = [double(true_candidate(:, 1)) disQueryTraining'];
    %sort the retrieval result in the order of the L2 distance to the query
    %data
%     tmp_list = sortrows(list, 2);

%     retrieved_list = [query_trueID; tmp_list(:, 1)];
    retrieved_list = [query_trueID; true_candidate(:, 1)];
    
    true_list=[query_trueID;exp_data.knn_p2(query_ID,1:numRetrieval)'];
    retrieved_list=[retrieved_list true_list];   
else
    queryDhamm = Dhamm(query_ID, :);
    
end

