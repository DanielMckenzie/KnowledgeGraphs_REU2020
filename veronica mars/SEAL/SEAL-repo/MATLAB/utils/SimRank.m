function [  auc, precision, sim ] = SimRank( train, test, lambda)
%% ����SimRankָ�겢����AUCֵ
    deg = sum(train,1);     
    % ��ڵ����ȣ�������������������
    lastsim = sparse(size(train,1), size(train,2)); 
    % �洢ǰһ���ĵ����������ʼ��Ϊȫ0����
    sim = sparse(eye(size(train,1))); 
    ntrain = train.*repmat(max(1./deg,0),size(train,1),1);
    % approximate SimRank
    for iter = 1:5
        sim = max(lambda*(ntrain'*sim*ntrain),eye(size(train,1)));
    end
    [auc, precision] = CalcAUC(train,test,sim);    
end
