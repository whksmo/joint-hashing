%% ---------------- independent test -------------------
mat = double(binary_test);
mat(mat==0)=-1;
corrletive_mat = zeros(48,48);
corrletive_mat = abs(mat * mat')/size(binary_test,2);

% for i = 1 : size(binary_test,1)
%     for j = i : size(binary_test,1)
%         sum(mat(i,:)==mat(j,:));
%         corrletive_mat(i,j) = ans;
%         corrletive_mat(j,i) = ans;
%     end
% end
% corrletive_mat = corrletive_mat/size(binary_test,2);

figure
 set (gcf,'unit','centimeters','Position',[10,4,7,5]);
% set(gca,'Position',[.2 .2 .7 .65]);
imshow(corrletive_mat,'InitialMagnification','fit');

clear mat corrletive_mat

%% balance test
mat = double(binary_test);
balance=sum(mat,2)/size(mat,2);
std_balance=std(balance);

%% correlance test
if(0)
    % bone
    test_file_list = './data/bone/test-file-list.txt';
    train_file_list = './data/bone/train-file-list.txt';
    test_label_file = './data/bone/test-label-bone.txt';
    train_label_file = './data/bone/train-label-bone.txt';
else
    % cifar10
    test_file_list = './data/cifar10/test-file-list.txt';
    test_label_file = './data/cifar10/test-label.txt';
    train_file_list = './data/cifar10/train-file-list.txt';
    train_label_file = './data/cifar10/train-label.txt';
end
trn_label = load(train_label_file);
tst_label = load(test_label_file);
% current preserve bits: rm_ind = [15,22,30,45,48  ,  1,3,23,28,29,31,35]
% rm_ind = find(map_list>=map);
% binary_test_ = binary_test;
binary_train_ = binary_train;
% binary_test_(rm_ind,:) = [];
binary_train_(rm_ind,:) = [];
map_list_trn=zeros(size(binary_train,1),1);
t=0;
[map_trn_lst, ~] = precision(trn_label, binary_train_, trn_label, binary_train_, top_k,1,0);
for i = 1 : size(binary_train,1)
    if(t<numel(rm_ind) && (i==rm_ind(t+1)))
        map_list_trn(i)=map_trn_lst;
        t=t+1;
    else        
%         temp_test=binary_test_;
        temp_train=binary_train_;
%         temp_test(i-t,:)=[];
        temp_train(i-t,:)=[];
        [map_, ~] = precision(trn_label, temp_train, trn_label, temp_train, top_k,1,0);
        map_list_trn(i)=map_;
        fprintf('%dth MAP = %f\n',i,map_);
    end
end
% MAP on test set after last removal
% [map_test, ~] = precision(trn_label, binary_train_, tst_label, binary_test_, top_k,1,0);

%% temp
figure
ax1 = subplot(2,1,1);
temp=[1:50];
set (gcf,'unit','centimeters','Position',[10,4,7,5]);
plot(ax1,[0,50],[map_trn_lst,map_trn_lst],1:length(map_list_trn),map_list_trn,temp(rm_ind),map_trn_lst*ones(numel(rm_ind)),'x');
ylabel('MAP\_train');
grid minor

ax2 = subplot(2,1,2);
set (gcf,'unit','centimeters','Position',[10,4,7,5]);
plot(ax2,[0,50],[map_tst_lst,map_tst_lst],1:length(map_list_tst),map_list_tst,temp(rm_ind),map_tst_lst*ones(numel(rm_ind)),'x');
ylabel('MAP\_test');
grid minor

%% map list test
% rm_ind = find(map_list>=map);
binary_test_ = binary_test;
binary_train_ = binary_train;
binary_test_(rm_ind,:) = [];
binary_train_(rm_ind,:) = [];
map_list_tst=zeros(size(binary_train,1),1);
t=0;
[map_tst_lst, ~] = precision(trn_label, binary_train_, tst_label, binary_test_, top_k,1,0);
for i = 1 : size(binary_train,1)
    if(t<numel(rm_ind) && (i==rm_ind(t+1)))
        map_list_tst(i)=map_tst_lst;
        t=t+1;
    else        
        temp_test=binary_test_;
        temp_train=binary_train_;
        temp_test(i-t,:)=[];
        temp_train(i-t,:)=[];
        [map_, ~] = precision(trn_label, temp_train, tst_label, temp_test, top_k,1,0);
        map_list_tst(i)=map_;
        fprintf('%dth MAP = %f\n',i,map_);
    end
end

%% add new bits
% open('./analysis/cifar10_modify/backup/0th_rm_comp.fig');
% obj=get(gca,'children');
% y1=get(obj(1),'ydata');
% y2=get(obj(2),'ydata');
norm_map_list = mapminmax([map_list_trn;0.99963426]',0,1)';
norm_map_list=norm_map_list-norm_map_list(end);
bit_weights =-ceil(norm_map_list/sum(abs(norm_map_list))*48);

modify_binary_tst=zeros(sum(bit_weights(bit_weights>0)),size(binary_test,2));
modify_binary_trn=zeros(sum(bit_weights(bit_weights>0)),size(binary_train,2));
ind=1;
for i = 1: numel(bit_weights)-1
    if(bit_weights(i)>0)
        for t = 1:bit_weights(i)
            modify_binary_tst(ind+t,:)=binary_test(i,:);
            modify_binary_trn(ind+t,:)=binary_train(i,:);
        end
        ind=ind+1;
    end
end

%% test modify 
[map_modify, ~] = precision(trn_label, modify_binary_trn, tst_label, modify_binary_tst, top_k,1,0);


%% plot precision && firing rate
% [B,I]=sort(map_list);
figure
ax1=subplot(1,1,1);
temp=[1:50];
set (gcf,'unit','centimeters','Position',[10,4,7,5]);
plot(ax1,[0,50],[map_new,map_new],1:length(map_list),map_list,temp(rm_ind),map_new*ones(numel(rm_ind)),'x');
ylabel('MAP');
grid minor
% fire rate  
% figure
% set (gcf,'unit','centimeters','Position',[10,4,7,5]);

% ax2=subplot(2,1,2);
% plot(ax2,[0,50],[0.5,0.5],1:length(balance),balance);
% ylim([0,1]);
% ylabel('fire rate');
% xlabel('hash bits');
% grid minor

clear temp
%% plot entropy comparison
figure
ax1=subplot(2,1,1);
set (gcf,'unit','centimeters','Position',[10,4,7,5]);
plot(ax1,1:length(map_list),abs(map_list-map));
ylabel('difference');
grid minor
% fire rate
% figure
% set (gcf,'unit','centimeters','Position',[10,4,7,5]);

ax2=subplot(2,1,2);
plot(ax2,1:length(balance),0.5-abs(balance-0.5));
ylim([0,0.5]);
ylabel('information');
xlabel('hash bits');
grid minor
