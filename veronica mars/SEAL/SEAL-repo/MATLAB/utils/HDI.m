function [ thisauc ] = HDI( train, test )
%% ����HDIָ�겢����AUCֵ
    sim = train * train;      
    % ��ɷ��ӵļ��㣬����ͬ��ͬ�ھ��㷨
    deg_row = repmat(sum(train,1), [size(train,1),1]);
    deg_row = deg_row .* spones(sim);
    deg_row = max(deg_row, deg_row');  
    % ��ɷ�ĸ�ļ��㣬����Ԫ��(i,j)��ʾȡ�˽ڵ�i�ͽڵ�j�Ķȵ����ֵ
    sim = sim ./ deg_row; clear deg_row;     
    % ������ƶȾ���ļ���
    sim(isnan(sim)) = 0; sim(isinf(sim)) = 0;
    thisauc = CalcAUC(train,test,sim, 10000);   
    % ���⣬�����ָ���Ӧ��AUC
end
