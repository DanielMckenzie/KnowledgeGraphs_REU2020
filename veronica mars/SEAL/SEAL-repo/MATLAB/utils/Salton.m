function [ thisauc ] = Salton( train, test )
%% ����Saltonָ�겢����AUCֵ
    tempdeg = repmat((sum(train,2)).^0.5,[1,size(train,1)]);       
    % �����������ģ��Ļ���Ҫ�ֿ顣
    tempdeg = tempdeg .* tempdeg';            
    % ��ĸ�ļ���
    sim = train * train;              
    % ���ӵļ���
    sim = sim./tempdeg;                 
    % ���ƶȾ���������
    sim(isnan(sim)) = 0; sim(isinf(sim)) = 0;
    thisauc = CalcAUC(train,test,sim, 10000);       
    % ���⣬�����ָ���Ӧ��AUC
end
