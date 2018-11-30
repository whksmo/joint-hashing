function exp_data = construct_data(db_info)
tic
fprintf('start constructing %s\n', [db_info.name, '_', db_info.type]);
if strcmp(db_info.name, 'cifar100') && strcmp(db_info.type, 'euclidean')
    dataset_path = '/data/images/cifar-100/';
    save_path = './data/cifar100/';
    num_tst = 10000;
    num_trn = 50000;
    if ~exist('./data/cifar100/cifar100_euclidean.mat','file')
        exp_data.db_name = db_info.name;
        exp_data.ft_type = db_info.type;
        test_data = zeros(num_tst,32,32,3);
        tst_file_list = textscan(fopen('./data/cifar100/val.txt'), '%s%d%d', 'delimiter', ' ');
        tst_label_list(:,1) = tst_file_list{1,2};
        tst_label_list(:,2) = tst_file_list{1,3};
        tst_file_list = tst_file_list{1,1};
        for i = 1 : num_tst
            img_name = tst_file_list(i);
            img_name = img_name{1};
            img = imread([dataset_path, img_name]);
            test_data(i, :, :, :) = img;
        end
        
        train_data = zeros(num_trn,32,32,3);
        trn_file_list = textscan(fopen('./data/cifar100/train.txt'), '%s%d%d', 'delimiter', ' ');
        trn_label_list(:,1) = trn_file_list{1,2};
        trn_label_list(:,2) = trn_file_list{1,3};
        trn_file_list = trn_file_list{1,1};
        for i = 1 : num_trn
            img_name = trn_file_list(i);
            img_name = img_name{1};
            img = imread([dataset_path, img_name]);
            train_data(i, :, :, :) = img;
        end
        test_data = uint8(reshape(test_data, num_tst, []));
        train_data = uint8(reshape(train_data, num_trn, []));
        total_data = [test_data; train_data];
        fprintf('computing euclidean feature for %s\n', db_info.name);
        
        Nneighbors = 0.02 * num_trn;
        DtrueTestTraining = distMat(test_data, train_data); % size = [Ntest x Ntraining]
        [Dball, I] = sort(DtrueTestTraining, 2); %sort columns
        exp_data.knn_p2 = uint32(I(:, 1:Nneighbors)); %take firsr 1000 nearest train datas compared to each test data
        exp_data.dis_p2 = Dball(:, 1:Nneighbors); %1000*1000
        exp_data.train_data = train_data;
        exp_data.test_data = test_data;
        exp_data.train_label = trn_label_list;
        exp_data.test_label = tst_label_list;
        save([save_path, db_info.name, '_', db_info.type], 'exp_data');
        fprintf('constructing %s database has finished\n', [db_info.name, '_', db_info.type]);
    else
        load ./data/cifar100/cifar100_euclidean exp_data
    end
elseif strcmp(db_info.name, 'cifar10') && strcmp(db_info.type, 'euclidean')
    dataset_path = './data/cifar10/';
    num_tst = 10000;
    num_trn = 50000;
    if ~exist('./data/cifar10/cifar10_euclidean.mat','file')
        exp_data.db_name = db_info.name;
        exp_data.ft_type = db_info.type;
        test_data = zeros(num_tst,32,32,3);
        tst_file_list = textscan(fopen('./data/cifar10/val.txt'), '%s%f', 'delimiter', ' ');
        tst_label_list = tst_file_list{1,2};
        tst_file_list = tst_file_list{1,1};
        for i = 1 : num_tst
            img_name = tst_file_list(i);
            img_name = img_name{1};
            img = imread([dataset_path, img_name]);
            test_data(i, :, :, :) = img;
        end
        
        train_data = zeros(num_trn,32,32,3);
        trn_file_list = textscan(fopen('./data/cifar10/train.txt'), '%s%f', 'delimiter', ' ');
        trn_label_list = trn_file_list{1,2};
        trn_file_list = trn_file_list{1,1};
        for i = 1 : num_trn
            img_name = trn_file_list(i);
            img_name = img_name{1};
            img = imread([dataset_path, img_name]);
            train_data(i, :, :, :) = img;
        end
        test_data = uint8(reshape(test_data, num_tst, []));
        train_data = uint8(reshape(train_data, num_trn, []));
        total_data = [test_data; train_data];
        fprintf('computing euclidean feature for %s\n', db_info.name);
        % for i = 1 : num_tst
        %     img = test_data(i,:,:,:);
        %     img = squeeze(img);
        %     euclidean_ft = get_features(img,[]);
        %     euclidean_tst(i,:) = euclidean_ft(:);
        % end
        % for i = 1 : num_trn
        %     img = train_data(i,:,:,:);
        %     img = squeeze(img);
        %     euclidean_ft = get_features(img,[]);
        %     euclidean_trn(i,:) = euclidean_ft(:);
        % end
        
        Nneighbors=0.02 * num_trn;
        DtrueTestTraining = distMat(test_data, train_data); % size = [Ntest x Ntraining]
        [Dball, I] = sort(DtrueTestTraining, 2); %sort columns
        exp_data.knn_p2 = uint32(I(:, 1:Nneighbors)); %take firsr 1000 nearest train datas compared to each test data
        exp_data.dis_p2 = Dball(:, 1:Nneighbors); %1000*1000
        exp_data.train_data = train_data;
        exp_data.test_data = test_data;
        exp_data.train_label = trn_label_list;
        exp_data.test_label = tst_label_list;
        save([dataset_path, db_info.name, '_', db_info.type], 'exp_data');
        fprintf('constructing %s database has finished\n', [db_info.name, '_', db_info.type]);
    else
        load ./data/cifar10/cifar10_euclidean exp_data
    end
elseif strcmp(db_info.name, 'cifar10') && strcmp(db_info.type, 'hog')
    dataset_path = './data/cifar10/';
    num_tst = 10000;
    num_trn = 50000;
    tic
    if ~exist('./data/cifar10/cifar10_hog.mat','file')
        
        test_data = zeros(num_tst,32,32,3);
        tst_file_list = textscan(fopen('./data/cifar10/val.txt'), '%s%f', 'delimiter', ' ');
        tst_label_list = tst_file_list{1,2};
        tst_file_list = tst_file_list{1,1};
        for i = 1 : num_tst
            img_name = tst_file_list(i);
            img_name = img_name{1};
            img = imread([dataset_path, img_name]);
            test_data(i, :, :, :) = img;
        end
        
        train_data = zeros(num_trn,32,32,3);
        trn_file_list = textscan(fopen('./data/cifar10/train.txt'), '%s%f', 'delimiter', ' ');
        trn_label_list = trn_file_list{1,2};
        trn_file_list = trn_file_list{1,1};
        for i = 1 : num_trn
            img_name = trn_file_list(i);
            img_name = img_name{1};
            img = imread([dataset_path, img_name]);
            train_data(i, :, :, :) = img;
        end
        test_data = uint8(test_data);
        train_data = uint8(train_data);
        fprintf('computing hog feature for %s\n', db_info.name);
        hog_tst = zeros(num_tst,8*8*42);
        hog_trn = zeros(num_trn,8*8*42);
        for i = 1 : num_tst
            img = test_data(i,:,:,:);
            img = squeeze(img);
            hog_ft = get_features(img,[]);
            hog_tst(i,:) = hog_ft(:);
        end
        for i = 1 : num_trn
            img = train_data(i,:,:,:);
            img = squeeze(img);
            hog_ft = get_features(img,[]);
            hog_trn(i,:) = hog_ft(:);
        end
        
        Nneighbors=0.02*num_trn;
        DtrueTestTraining = distMat(hog_tst,hog_trn); % size = [Ntest x Ntraining]
        [Dball, I] = sort(DtrueTestTraining,2); %sort columns
        exp_data.knn_p2 = I(:,1:Nneighbors); %take firsr 1000 nearest train datas compared to each test data
        exp_data.dis_p2 = Dball(:,1:Nneighbors); %1000*1000
        exp_data.train_data = hog_trn;
        exp_data.test_data = hog_tst;
        exp_data.train_label = trn_label_list;
        exp_data.test_label = tst_label_list;
        exp_data.db_name = db_info.name;
        exp_data.ft_type = db_info.type;
        save([dataset_path, db_info.name, '_', db_info.type], 'exp_data');
        fprintf('constructing %s database has finished\n', [db_info.name, '_', db_info.type]);
    else
        load ./data/cifar10/cifar10_hog exp_data
    end
elseif strcmp(db_info.name, 'cifar10') && strcmp(db_info.type, 'ssim')
    dataset_path = './data/cifar10/';
    num_tst = 10000;
    num_trn = 50000;
    tic
    if ~exist('./data/cifar10/cifar10_ssim.mat','file')
        
        test_data = zeros(num_tst,32,32,3);
        tst_file_list = textscan(fopen('./data/cifar10/val.txt'), '%s%f', 'delimiter', ' ');
        tst_label_list = tst_file_list{1,2};
        tst_file_list = tst_file_list{1,1};
        for i = 1 : num_tst
            img_name = tst_file_list(i);
            img_name = img_name{1};
            img = imread([dataset_path, img_name]);
            test_data(i, :, :, :) = img;
        end
        
        train_data = zeros(num_trn,32,32,3);
        trn_file_list = textscan(fopen('./data/cifar10/train.txt'), '%s%f', 'delimiter', ' ');
        trn_label_list = trn_file_list{1,2};
        trn_file_list = trn_file_list{1,1};
        for i = 1 : num_trn
            img_name = trn_file_list(i);
            img_name = img_name{1};
            img = imread([dataset_path, img_name]);
            train_data(i, :, :, :) = img;
        end
        test_data = uint8(test_data);
        train_data = uint8(train_data);
        fprintf('start computing ssim feature for %s\n', db_info.name);
        ssim_mat = zeros(num_tst, num_trn);
        % parpool(10);
        for i = 1 : num_tst
            img_tst = test_data(i,:,:,:);
            img_tst = squeeze(img_tst);
            parfor j = 1 : num_trn
                img_trn = train_data(j,:,:,:);
                img_trn = squeeze(img_trn);
                ssim_mat(i,j) = ssim(img_tst, img_trn);
            end
            toc
        end
        
        Nneighbors=0.02*num_trn;
        [Dball, I] = sort(ssim_mat,2); %sort columns
        exp_data.knn_p2 = I(:,1:Nneighbors); %take firsr 1000 nearest train datas compared to each test data
        exp_data.dis_p2 = Dball(:,1:Nneighbors); %1000*1000
        exp_data.train_label = trn_label_list;
        exp_data.test_label = tst_label_list;
        exp_data.train_data = train_data;
        exp_data.test_data = test_data;
        exp_data.db_name = db_info.name;
        exp_data.ft_type = db_info.type;
        save([dataset_path, db_info.name, '_', db_info.type], 'exp_data');
        fprintf('constructing %s database has finished\n', [db_info.name, '_', db_info.type]);
    else
        load ./data/cifar10/cifar10_ssim exp_data
    end
end
toc
end

