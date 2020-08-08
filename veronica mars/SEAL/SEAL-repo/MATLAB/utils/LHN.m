function [ thisauc ] = LHN( train, test )
%% ����LHN1ָ�겢����AUCֵ
    sim = train * train;     
    % ��ɷ��ӵļ��㣬����ͬ��ͬ�ھ��㷨
    deg = sum(train,2);
    deg = deg*deg';                                         
    %��ɷ�ĸ�ļ���
    sim = sim ./ deg;                                      
    %���ƶȾ���ļ���
    sim(isnan(sim)) = 0; sim(isinf(sim)) = 0;
    thisauc = CalcAUC(train,test,sim, 10000);  
    % ���⣬�����ָ���Ӧ��AUC
end
