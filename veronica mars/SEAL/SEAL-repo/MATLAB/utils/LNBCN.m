function [ thisauc ] = LNBCN( train, test ) 
%% ����ֲ����ر�Ҷ˹ģ����CNָ�겢����AUCֵ
    s = size(train,1)*(size(train,1)-1) / nnz(train) -1;  
    % ����ÿ�������еĳ���s
    tri = diag(train*train*train)/2;     
    % ����ÿ�������ڵ������θ���
    tri_max = sum(train,2).*(sum(train,2)-1)/2;  
    % ÿ�������������ڵ������θ���
    R_w = (tri+1)./(tri_max+1); clear tri tri_max; 
    % �����������ǰ��չ�ʽ����ÿ����Ľ�ɫ  
    SR_w = log(s)+log(R_w); clear s R_w;
    SR_w(isnan(SR_w)) = 0; SR_w(isinf(SR_w)) = 0;
    SR_w = repmat(SR_w,[1,size(train,1)]) .* train;   
    % �ڵ�Ľ�ɫ�������
    sim = spones(train) * SR_w;   clear SR_w;                       
    % ���ڵ�ԣ�x,y���Ĺ�ͬ�ھӵĽ�ɫ����ֵ��Ӽ���
    thisauc = CalcAUC(train,test,sim, 10000);
    % ���⣬�����ָ���Ӧ��AUC
end
