function [ auc, precision, sim ] = CN( train, test )
%% ����CNָ�겢����AUCֵ
    sim = train * train;        
    % ���ƶȾ���ļ���
    [auc, precision] = CalcAUC(train,test,sim);
    % ���⣬�����ָ���Ӧ��AUC
end
