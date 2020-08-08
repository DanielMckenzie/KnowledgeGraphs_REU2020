function [ thisauc ] = TSCN( train, test, lambda )
%% ����TSCNָ�겢����AUCֵ
    sim = train * train;                       
    % ���㹲ͬ�ھ����ƶȾ���
    I = sparse(eye(size(train,1)));
    sim = inv(I - lambda*sim) * sim;
    % ���ƶ�ת��
    thisauc = CalcAUC(train,test,sim, 10000);     
    % ���⣬�����ָ���Ӧ��AUC
end
