function [ auc, precision, sim ] = PA( train, test )
%% ����PAָ�겢����AUCֵ
    deg_row = sum(train,2);       
    % ���нڵ�Ķȹ�����������������������ת�ü���
    sim = deg_row * deg_row';  
    clear deg_row deg_col;       
    % ���ƶȾ���������
    [auc, precision] = CalcAUC(train,test,sim); 
    % ���⣬�����ָ���Ӧ��AUC
end
