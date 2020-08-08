function [ auc, precision, sim ] = Katz( train, test, lambda )
%% ����katzָ�겢����AUCֵ
    sim = inv( sparse(eye(size(train,1))) - lambda * train);   
    % �����Ծ���ļ���
    sim = sim - sparse(eye(size(train,1)));
    [auc, precision] = CalcAUC(train,test,sim);   
    % ���⣬�����ָ���Ӧ��AUC
end
